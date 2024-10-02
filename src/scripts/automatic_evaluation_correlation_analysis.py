from src.logic.types import AuthorResult
from src.extraction.evaluation import tournament_ranking
from src.scripts.expert_evaluation_analysis import get_all_automatic_jsons, get_queries_from_evaluation_folder

if __name__ == '__main__':
    results = [AuthorResult.from_json(data) for data in get_all_automatic_jsons()]
    queries, emails = get_queries_from_evaluation_folder('evaluation/_DONE')
    EVALUATION_MODEL = 'dev-llama-3-large'  # 'alias-large-instruct'  #

    # evaluate self consistency of the leafes
    missmatches = 0
    profile_1_preferred = 0
    times_missmatched_and_profile_1_preferred = 0
    times_missmatched_and_profile_2_preferred = 0
    evaluations = 0
    for result in results:
        query = queries[result.author]
        for node in result.tournament.all_leafes:
            if node.match.profiles[0] == node.match.profiles[1]:
                continue
            print(f'Evaluating {result.author} for {node.match.profiles}')

            res1 = tournament_ranking(
                EVALUATION_MODEL,
                query,
                {
                    node.match.profiles[0]: result.profiles[node.match.profiles[0]],
                    node.match.profiles[1]: result.profiles[node.match.profiles[1]],
                },
                do_shuffle=False,
            )

            res2 = tournament_ranking(
                EVALUATION_MODEL,
                query,
                {
                    node.match.profiles[1]: result.profiles[node.match.profiles[1]],
                    node.match.profiles[0]: result.profiles[node.match.profiles[0]],
                },
                do_shuffle=False,
            )

            print('Res1:', res1.match.winner, 'Res2:', res2.match.winner)
            print('Reasoning1:', res1.match.reasoning)
            print('Reasoning2:', res2.match.reasoning)
            print('\n\n\n\n')

            profile_1_preferred += res1.match.preferred_profile_index == 0
            profile_1_preferred += res2.match.preferred_profile_index == 0
            missmatches += res1.match.winner != res2.match.winner
            if res1.match.winner != res2.match.winner:
                if res1.match.preferred_profile_index == 0:
                    times_missmatched_and_profile_1_preferred += 1
                else:
                    times_missmatched_and_profile_2_preferred += 1
                if res2.match.preferred_profile_index == 0:
                    times_missmatched_and_profile_1_preferred += 1
                else:
                    times_missmatched_and_profile_2_preferred += 1
            evaluations += 1

    print(f'Missmatches: {missmatches} / {evaluations} ({missmatches / evaluations * 100:.2f}%)')
    print(
        f'Profile 1 preferred: {profile_1_preferred} / {evaluations * 2} ({profile_1_preferred / evaluations * 50:.2f}%)'
    )
    print(
        f'Times missmatched and profile 1 preferred: {times_missmatched_and_profile_1_preferred} / {missmatches * 2} ({times_missmatched_and_profile_1_preferred / missmatches * 50:.2f}%)'
    )
    print(
        f'Times missmatched and profile 2 preferred: {times_missmatched_and_profile_2_preferred} / {missmatches * 2} ({times_missmatched_and_profile_2_preferred / missmatches * 50:.2f}%)'
    )
