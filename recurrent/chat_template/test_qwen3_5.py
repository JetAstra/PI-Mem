import pytest
from transformers import AutoTokenizer

from recurrent.chat_template.utils import set_chat_template


MODEL_PATH = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"

TEST_MESSAGES = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice."},
    {"role": "tool", "content": "I love fresh lemon juice."},
]

TEST_MESSAGES_NOT_FINISH = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {
        "role": "assistant",
        "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice.",
        "finished": False,
    },
]


@pytest.fixture
def tokenizers():
    tokenizer_org = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    set_chat_template(tokenizer)
    return {
        "tokenizer_org": tokenizer_org,
        "tokenizer": tokenizer,
    }


def test_tokenizer_type(tokenizers):
    tokenizer_org = tokenizers["tokenizer_org"]
    assert type(tokenizer_org).__name__ == "TokenizersBackend", (
        f"unexpected tokenizer type: {type(tokenizer_org)}"
    )


def test_template_injected(tokenizers):
    tokenizer_org = tokenizers["tokenizer_org"]
    tokenizer = tokenizers["tokenizer"]
    assert "{% generation %}" not in (tokenizer_org.chat_template or "")
    assert "{% generation %}" in (tokenizer.chat_template or "")


def test_message_string_consistency(tokenizers):
    tokenizer_org = tokenizers["tokenizer_org"]
    tokenizer = tokenizers["tokenizer"]

    msg_str_org = tokenizer_org.apply_chat_template(
        TEST_MESSAGES, return_dict=False, tokenize=False
    ).rstrip()

    msg_str = tokenizer.apply_chat_template(
        TEST_MESSAGES, return_dict=False, tokenize=False
    ).rstrip()

    assert msg_str == msg_str_org, f"\nOriginal: {repr(msg_str_org)}\nCustom: {repr(msg_str)}"


def test_token_consistency(tokenizers):
    tokenizer_org = tokenizers["tokenizer_org"]
    tokenizer = tokenizers["tokenizer"]

    tokens_org = tokenizer_org.apply_chat_template(
        TEST_MESSAGES,
        return_dict=True,
        return_assistant_tokens_mask=False,
        return_tensors="pt",
    )

    tokens = tokenizer.apply_chat_template(
        TEST_MESSAGES,
        return_dict=True,
        return_assistant_tokens_mask=False,
        return_tensors="pt",
    )

    assert (tokens.input_ids == tokens_org.input_ids).all(), (
        f"Token mismatch:\nCustom: {tokens.input_ids}\nOriginal: {tokens_org.input_ids}"
    )


def test_assistant_reply(tokenizers):
    tokenizer = tokenizers["tokenizer"]

    tokens = tokenizer.apply_chat_template(
        TEST_MESSAGES, return_dict=True, return_assistant_tokens_mask=True, return_tensors="pt"
    )

    left = tokenizer.decode(tokens["input_ids"][tokens["assistant_masks"].bool()])
    assert TEST_MESSAGES[1]["content"] in left, f"\nDecoded: {repr(left)}"
    assert tokenizer.eos_token in left, f"\nDecoded: {repr(left)}"


def test_finish(tokenizers):
    tokenizer = tokenizers["tokenizer"]
    tokens = tokenizer.apply_chat_template(
        TEST_MESSAGES_NOT_FINISH, return_dict=False, tokenize=False
    )
    assert not tokens.rstrip().endswith(tokenizer.eos_token), tokens


@pytest.mark.parametrize(
    "enable_thinking, expected_suffix",
    [
        (False, "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        (True, "<|im_start|>assistant\n<think>\n"),
    ],
)
def test_apply_chat_template_kwargs_enable_thinking(tokenizers, enable_thinking, expected_suffix):
    tokenizer = tokenizers["tokenizer"]
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        return_dict=False,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    assert rendered.endswith(expected_suffix), (
        f"Unexpected suffix for enable_thinking={enable_thinking}:\n{repr(rendered[-120:])}"
    )


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_generation_prompt_think_scaffold_not_in_assistant_mask(tokenizers, enable_thinking):
    tokenizer = tokenizers["tokenizer"]
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
        enable_thinking=enable_thinking,
    )
    assert int(tokens["assistant_masks"].sum().item()) == 0


def test_thinking_false_think_tags_are_prompt_only(tokenizers):
    tokenizer = tokenizers["tokenizer"]
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "ans"}],
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
    )
    masked = tokenizer.decode(tokens["input_ids"][tokens["assistant_masks"].bool()])
    assert "<think>" not in masked
    assert "</think>" not in masked
    assert "ans" in masked


def test_thinking_true_opening_think_is_prompt_closing_think_is_generation(tokenizers):
    tokenizer = tokenizers["tokenizer"]
    tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "<think>\nreasoning\n</think>\n\nfinal"},
        ],
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
    )
    masked = tokenizer.decode(tokens["input_ids"][tokens["assistant_masks"].bool()])
    assert "<think>" not in masked
    assert "</think>" in masked
    assert "reasoning" in masked
    assert "final" in masked




from transformers import AutoTokenizer
from recurrent.chat_template.utils import set_chat_template


p='/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307'
t=AutoTokenizer.from_pretrained(p, trust_remote_code=True)
set_chat_template(t)


# case1: enable_thinking=False style (assistant content has no think tags)
msgs1=[{'role':'user','content':'u'},{'role':'assistant','content':'ans'}]
o1=t.apply_chat_template(msgs1, return_dict=True, return_tensors='pt', return_assistant_tokens_mask=True)
masked1=t.decode(o1['input_ids'][0][o1['assistant_masks'][0].bool()])
full1=t.decode(o1['input_ids'][0])
print('CASE1 masked:',repr(masked1))
print('CASE1 has <think> in masked?', '<think>' in masked1, '</think>' in masked1)
print('CASE1 full tail:',repr(full1[-120:]))


# case2: enable_thinking=True style (assistant content includes think close)
msgs2=[{'role':'user','content':'u'},{'role':'assistant','content':'<think>\nreasoning\n</think>\n\nfinal'}]
o2=t.apply_chat_template(msgs2, return_dict=True, return_tensors='pt', return_assistant_tokens_mask=True)
masked2=t.decode(o2['input_ids'][0][o2['assistant_masks'][0].bool()])
full2=t.decode(o2['input_ids'][0])
print('CASE2 masked:',repr(masked2))
print('CASE2 has <think> in masked?', '<think>' in masked2, '</think>' in masked2)
print('CASE2 full tail:',repr(full2[-120:]))