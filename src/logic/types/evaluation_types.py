from __future__ import annotations


from dataclasses import dataclass, field


from src.logic.types.base_types import Profile


@dataclass(frozen=True)
class ExtractedProfile:
    profile: Profile
    model: str  # Identifier from OpenAI/Insomnium
    number_of_examples: int
    extraction_function: str
    extraction_time: float

    @staticmethod
    def from_profile(profile: Profile) -> ExtractedProfile:
        return ExtractedProfile(
            profile=profile,
            model='None',
            number_of_examples=0,
            extraction_function='None',
            extraction_time=0.0,
        )

    @staticmethod
    def from_json(data: dict) -> ExtractedProfile:
        return ExtractedProfile(
            profile=Profile.from_json(data['profile']),
            model=data['model'],
            number_of_examples=data['number_of_examples'],
            extraction_function=data['extraction_function'],
            extraction_time=data['extraction_time'],
        )


@dataclass(frozen=True)
class RankingResult:
    # Represents a 2way ranking between two profiles
    profiles: tuple[int, int]
    preferred_profile_index: int  # (0 or 1) The index of the preferred profile in the profiles tuple
    reasoning: str | None  # let the model generate the reasoning before returning the preferred profile

    @property
    def winner(self) -> int:
        return self.profiles[self.preferred_profile_index]

    @property
    def loser(self) -> int:
        return self.profiles[1 - self.preferred_profile_index]


@dataclass(frozen=True)
class TournamentNode:
    match: RankingResult
    children: list[TournamentNode] = field(default_factory=list)

    @property
    def all_nodes(self) -> list[TournamentNode]:
        nodes: list[TournamentNode] = [self]
        for child in self.children:
            nodes.extend(child.all_nodes)
        return nodes

    @property
    def all_leafes(self) -> list[TournamentNode]:
        leafes: list[TournamentNode] = []
        for child in self.children:
            leafes.extend(child.all_leafes)
        if not self.children:
            leafes.append(self)
        return leafes

    @property
    def all_profiles_in_loser_subtree(self) -> list[int]:
        if not self.children:
            return [self.match.loser]

        if len(self.children) != 2:
            return []

        loser_child = self.children[1 - self.match.preferred_profile_index]
        profiles: list[int] = []
        for node in loser_child.all_leafes:
            for profile in node.match.profiles:
                profiles.append(profile)
        return profiles


@dataclass(frozen=True)
class AuthorResult:
    tournament: TournamentNode
    profiles: dict[int, ExtractedProfile]
    titles: list[str]
    author: str

    @staticmethod
    def from_json(data: dict) -> AuthorResult:
        tournament, profiles = parse_tournament_and_profiles_from_json(data)
        return AuthorResult(
            tournament=tournament,
            profiles=profiles,
            titles=data['titles'],
            author=data['author'],
        )


def parse_tournament_and_profiles_from_json(data: dict) -> tuple[TournamentNode, dict[int, ExtractedProfile]]:
    # convert tournament dictionary to TournamentNode
    def process_node(node_data: dict) -> TournamentNode:
        node = TournamentNode(
            match=RankingResult(
                profiles=node_data['match']['profiles'],
                preferred_profile_index=node_data['match']['preferred_profile_index'],
                reasoning=node_data['match'].get('reasoning', None),
            )
        )
        for child_data in node_data['children']:
            node.children.append(process_node(child_data))
        return node

    tournament = process_node(data['tournament'])

    # convert profiles dictionary to dict[int, ExtractedProfile]
    profiles = {}
    for key, profile_data in data['profiles'].items():
        profiles[int(key)] = ExtractedProfile(
            profile=Profile.from_json(profile_data['profile']),
            model=profile_data['model'],
            number_of_examples=profile_data['number_of_examples'],
            extraction_function=profile_data['extraction_function'],
            extraction_time=profile_data['extraction_time'],
        )

    return tournament, profiles
