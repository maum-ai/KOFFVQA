import transformers
from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel

from .utils import *


base_judge_prompt = '''
어떤 사진에 대한 질문과 답변, 그리고 채점 기준이 다음과 같이 주어질 때, 주어진 채점 기준 항목을 기준으로 답변이 얼마나 옳은 답변인지 채점을 한 결과가 필요합니다.

사진은 주어지지 않지만, 아래 주어질 채점 기준을 바탕으로 어떤 내용인지 추정이 가능합니다. 채점 기준을 기반으로만 채점해 주세요.

채점의 목적은 주어진 답변이 사람이 보기에 얼마나 올바르고 질이 좋은 답변인지 판단하는 것입니다. 이것을 고려해서 채점해 주세요.

질문은 아래와 같습니다.
> "{0}"

그리고 이 질문에 대한 답변은 아래와 같습니다.
> "{1}"

위의 답변을 채점할 채점 기준은 다음과 같습니다.

{2}

위 기준을 바탕으로 답변을 10점 만점의 점수로 채점하세요. 단, 채점 기준의 각 항목을 부분적으로만 만족하는 경우에도 부분 점수는 없습니다. 무조건 0에서 10 사이의 점수로 채점해야 합니다.

단, 유의어 또는 유사한 표현을 사용한 설명은 말이 정확히 같지 않더라도 옳을 수 있습니다. 단, 여기에 예외가 있는데, 사진에 있는 글을 그대로 읽어야 하는 질문의 경우에는 유의어라도 정확히 일치하지 않게 잘못 읽었다면 틀린 답변입니다.

먼저 각 채점 기준에 대한 판단을 나열하고, 채점 결과를 종합하여 맨 마지막 줄에 "최종 점수: 10점 만점에 X점"의 형식으로 최종 점수를 쓰세요.
'''.strip()


class base_judge(object):
    '''
    Base class for judge LLM.
    '''
    def __init__(self, arc_class, judgepath:str, gen_kwargs:dict={}, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(judgepath, trust_remote_code=True)
        if isinstance(arc_class, str):
            arc_class = getattr(transformers, arc_class)
        device = 'auto' if kwargs.get('auto_device', False) else 'cuda'
        model = arc_class.from_pretrained(judgepath, device_map=device, torch_dtype='auto', attn_implementation="flash_attention_2", trust_remote_code=True)
        self.model = model.eval()
        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs

    def generate(self, prompt:str, **kwargs) -> str:
        messages = [{'role': 'user', 'content': prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True).to('cuda')
        output = self.model.generate(**input_ids, **self.gen_kwargs)
        return self.tokenizer.decode(output[0][input_ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def auto_judge(judgepath:str, **kwargs) -> base_judge:
    '''
    Automatically creates a judge based on the given paths.
    '''
    supported_judges = {
        'google/gemma-2-9b-it': dict(use_cache=True, max_new_tokens=1024, do_sample=False),
    }
    if judgepath in supported_judges:
        return base_judge(AutoModelForCausalLM, judgepath, supported_judges[judgepath])
    raise ValueError(f'This judge ({judgepath}) cannot be used for evaluation.')
