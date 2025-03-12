from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

@task
def tiny_mmlu():
    return Task(
        dataset=hf_dataset(
            path="tinyBenchmarks/tinyMMLU",
            split="test",
            sample_fields=lambda record: Sample(
                input=record["question"],
                choices=record["choices"],
                target="ABCD"[record["answer"]],
            )
        ),
        solver=multiple_choice(),
        scorer=choice(),
    )
