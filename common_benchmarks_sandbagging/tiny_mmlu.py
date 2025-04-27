from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


def load_tiny_mmlu_dataset():
    return hf_dataset(
        path="tinyBenchmarks/tinyMMLU",
        split="test",
        sample_fields=lambda record: Sample(
            input=record["question"],
            choices=record["choices"],
            target="ABCD"[record["answer"]],
        ),
    )


@task
def tiny_mmlu():
    return Task(
        dataset=load_tiny_mmlu_dataset(),
        solver=multiple_choice(cache_prompt=True),
        scorer=choice(),
    )
