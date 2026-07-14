"""Prompt formatting helpers."""


def user_prompt(tokenizer, question):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User:\n{question}\n\nAssistant:\n"


def supervised_text(tokenizer, question, answer):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
    eos = tokenizer.eos_token or ""
    return f"User:\n{question}\n\nAssistant:\n{answer}{eos}"
