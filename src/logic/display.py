import os
import json
import urllib.parse

from src.util.log import date_str
from src.logic.types import AuthorResult, Message
from src.util import custom_asdict


def write_author_result(author_result: AuthorResult, email: str, output_folder: str = 'results'):
    dump_author_result_to_json(author_result, output_folder)
    generate_html_file_for_tournament_evaluation(author_result, output_folder)
    generate_html_file_for_tournament_ranking_result(author_result, output_folder)
    generate_mail_for_author_result(author_result, email, output_folder)
    generate_follow_up_mail_for_author_result(author_result, email, output_folder)


def generate_mail_for_author_result(result: AuthorResult, email: str, output_folder: str = 'results') -> None:
    MAIL_TEMPLATE = """
**Email:** [[EMAIL]](mailto:[EMAIL])  
**Subject:** Request for Participation in Competency Profile Evaluation

---  
    
Dear [NAME],

I hope this mail finds you well. As part of my Master's thesis and the research project [Kompetenznetzwerk](https://bis.aifb.kit.edu/317_389.php) of the KIT, I am conducting a study to evaluate the accuracy of various methods for automatically extracting competency profiles from research papers. Your expertise and feedback would be immensely valuable to this research.

The evaluation involves comparing personalized competency profiles that have been generated based on your five most cited papers. Since these profiles are derived from your own work, your input is crucial in determining which profile best reflects your competencies. The process should take no more than five minutes of your time, and detailed instructions are provided directly on the webpage to guide you through each step.

To participate, simply click on the link below to access the evaluation page:

**[Start the Evaluation]([LINK])**

Your participation is greatly appreciated and will contribute significantly to the development of more accurate competency extraction methods.

Thank you very much for your time and assistance!

Best regards,  
Bertil Braun  

KIT - Karlsruhe Institute of Technology  
[bertil.braun@student.kit.edu](mailto:bertil.braun@student.kit.edu)  
+49 1525 3810140

---

Link to the evaluation page: [LINK]  
Link to the Research Project Kompetenznetzwerk: https://bis.aifb.kit.edu/317_389.php"""

    link = f'https://evaluation.tiiny.site/{result.author.replace(" ", "%20")}/{result.author.replace(" ", "%20")}.evaluation.html'

    mail = MAIL_TEMPLATE.replace('[NAME]', result.author).replace('[LINK]', link).replace('[EMAIL]', email)

    base_url = 'https://www.digitalocean.com/community/markdown#?md='
    encoded_content = urllib.parse.quote(mail)

    render_link = base_url + encoded_content

    with open(f'{output_folder}/{result.author}.mail.txt', 'w') as f:
        f.write(f'Render link: {render_link}\n\n\n\n')
        f.write(mail)


def generate_follow_up_mail_for_author_result(result: AuthorResult, email: str, output_folder: str = 'results') -> None:
    MAIL_TEMPLATE = """
**Email:** [[EMAIL]](mailto:[EMAIL])  
**Subject:** Reminder: Competency Profile Evaluation for Research Project

---  
    
Dear [NAME],


I hope you're doing well. I wanted to send a quick follow-up regarding the evaluation for my Master’s thesis, which is part of the research project [Kompetenznetzwerk](https://bis.aifb.kit.edu/317_389.php) at KIT.

If you haven’t had the chance yet to complete the evaluation, I would greatly appreciate your participation. It involves reviewing personalized competency profiles generated from your five most cited papers and should take no more than five minutes. Your feedback is crucial to the success of this research.

The evaluation period ends this Friday, and your input would be extremely valuable.

**[Start the Evaluation]([LINK])**

Thank you very much for your time and consideration. Please feel free to reach out if you have any questions.

Best regards,  
Bertil Braun  

KIT - Karlsruhe Institute of Technology  
[bertil.braun@student.kit.edu](mailto:bertil.braun@student.kit.edu)  
+49 1525 3810140

---

Link to the evaluation page: [LINK]  
Link to the Research Project Kompetenznetzwerk: https://bis.aifb.kit.edu/317_389.php"""

    link = f'https://evaluation.tiiny.site/{result.author.replace(" ", "%20")}/{result.author.replace(" ", "%20")}.evaluation.html'

    mail = MAIL_TEMPLATE.replace('[NAME]', result.author).replace('[LINK]', link).replace('[EMAIL]', email)

    base_url = 'https://www.digitalocean.com/community/markdown#?md='
    encoded_content = urllib.parse.quote(mail)

    render_link = base_url + encoded_content

    with open(f'{output_folder}/{result.author}.follow_up_mail.txt', 'w') as f:
        f.write(f'Render link: {render_link}\n\n\n\n')
        f.write(mail)


def generate_html_file_for_tournament_evaluation(author_result: AuthorResult, output_folder: str = 'results'):
    json_data = json.dumps(custom_asdict(author_result), indent=4)
    with open('src/template/template_tournament_evaluation.html', 'r') as file:
        html_template = file.read()

    html_content = html_template.replace('"{{authorData}}"', json_data)

    output_file_path = os.path.abspath(f'{output_folder}/{author_result.author}.evaluation.html')
    _write_and_display(html_content, output_file_path)


def generate_html_file_for_tournament_ranking_result(author_result: AuthorResult, output_folder: str = 'results'):
    json_data = json.dumps(custom_asdict(author_result), indent=4)
    with open('src/template/template_tournament_ranking_result.html', 'r') as file:
        html_template = file.read()

    html_content = html_template.replace('"{{authorData}}"', json_data)

    output_file_path = os.path.abspath(f'{output_folder}/{author_result.author}.tournament.html')
    _write_and_display(html_content, output_file_path)


def dump_author_result_to_json(author_result: AuthorResult, output_folder: str = 'results'):
    output_file_path = os.path.abspath(f'{output_folder}/{author_result.author}.json')
    with open(output_file_path, 'w') as file:
        json.dump(custom_asdict(author_result), file, indent=4)


def generate_html_file_for_chat(messages: list[Message], chat_name: str = 'chat'):
    json_data = json.dumps([message.to_dict() for message in messages], indent=4)
    with open('src/template/template_chat.html', 'r') as file:
        html_template = file.read()

    html_content = html_template.replace('"{{chatData}}"', json_data).replace('"{{fileName}}"', chat_name)

    output_file_path = os.path.abspath(f'logs/{date_str()}/chat_{chat_name}.html')
    _write_and_display(html_content, output_file_path)


def _write_and_display(html_content: str, output_file_path: str):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as file:
        file.write(html_content)

    print('file:///' + output_file_path.replace('\\', '/'))


if __name__ == '__main__':
    from src.logic.types import (
        Competency,
        Profile,
        ExtractedProfile,
        SystemMessage,
        HumanMessage,
        AIMessage,
        AIExampleMessage,
        HumanExampleMessage,
        TournamentNode,
        RankingResult,
    )

    competencies = [
        Competency('AI', 'Study of artificial intelligence'),
        Competency('ML', 'Machine learning algorithms'),
    ]
    profile = Profile('Technology', competencies)
    extracted_profile = ExtractedProfile(
        profile,
        model='OpenAI GPT-4',
        number_of_examples=2,
        extraction_function='extraction_function',
        extraction_time=0.5,
    )
    ranking_result = RankingResult(profiles=(1, 1), reasoning='Same profile', preferred_profile_index=0)
    author_result = AuthorResult(
        TournamentNode(match=ranking_result, children=[]),
        {1: extracted_profile},
        ['Paper on AI', 'Thesis on ML'],
        'John Doe',
    )

    generate_html_file_for_tournament_evaluation(author_result)

    generate_html_file_for_tournament_ranking_result(author_result)

    generate_mail_for_author_result(author_result, 'email@mail.com')
    generate_follow_up_mail_for_author_result(author_result, 'email@mail.com')

    messages = [
        SystemMessage('Welcome to the Chat!'),
        HumanExampleMessage('I am a student'),
        AIExampleMessage('Hello student!'),
        HumanMessage('Hello!'),
        AIMessage('Hi, how can I help you today?'),
        HumanMessage('I need help with my homework'),
        AIMessage('Sure! What subject is it?'),
        HumanMessage('It is about AI and ML'),
        AIMessage('Great! I can help you with that!'),
    ]

    generate_html_file_for_chat(messages)
