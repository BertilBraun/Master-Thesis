from dataclasses import dataclass
from src.logic.types import Example, Profile


@dataclass
class SampleForFineTuningImprovementEvaluation:
    prompt: str
    abstracts: list[str]
    best_profile_from_original_model: str
    best_profile_from_second_to_last_model: str
    best_profile_from_last_model: str

    @staticmethod
    def from_json(data: dict) -> 'SampleForFineTuningImprovementEvaluation':
        if 'best_profile_from_second_to_last_model' not in data:
            data['best_profile_from_second_to_last_model'] = data['best_profile_from_last_model']
        return SampleForFineTuningImprovementEvaluation(**data)

    def with_new_profiles(
        self,
        best_profile_from_last_model: str,
        best_profile_from_original_model: str | None = None,
    ) -> 'SampleForFineTuningImprovementEvaluation':
        return SampleForFineTuningImprovementEvaluation(
            prompt=self.prompt,
            abstracts=self.abstracts,
            best_profile_from_original_model=best_profile_from_original_model or self.best_profile_from_original_model,
            best_profile_from_last_model=best_profile_from_last_model,
            best_profile_from_second_to_last_model=self.best_profile_from_last_model,
        )


@dataclass(frozen=True)
class SampleToGenerate:
    author: str
    abstracts: list[str]
    examples: list[Example]

    @staticmethod
    def from_json(data: dict) -> 'SampleToGenerate':
        return SampleToGenerate(
            author=data['author'],
            abstracts=data['abstracts'],
            examples=[Example.from_json(example) for example in data['examples']],
        )


@dataclass(frozen=True)
class SampleToEvaluate:
    author: str
    prompt: str
    abstracts: list[str]
    profiles: list[Profile]

    @staticmethod
    def from_json(data: dict) -> 'SampleToEvaluate':
        return SampleToEvaluate(
            author=data['author'],
            prompt=data['prompt'],
            abstracts=data['abstracts'],
            profiles=[Profile.from_json(profile) for profile in data['profiles']],
        )


@dataclass(frozen=True)
class PreferenceSample:
    # WARNING there is a copy of this class in src/finetuning/train.py
    prompt: str
    chosen: str
    rejected: str

    @staticmethod
    def from_json(data: dict) -> 'PreferenceSample':
        return PreferenceSample(**data)
