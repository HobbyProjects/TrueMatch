import streamlit as st
import os
import pickle
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import random

from azure.storage.blob import BlobServiceClient

container_name = 'userprofies'

class OpenAIAgent:
    def __init__(self, llm, system_prompt: str):
        self._llm = llm
        self._system_prompt: str = system_prompt
        prompt_template = PromptTemplate(input_variables=["context", "topic"], template=self._system_prompt)

        self._llm_chain = LLMChain(
            prompt=prompt_template,
            llm=self._llm,
            verbose=True
        )

    def ask(self, user_prompt: str, context) -> str:
        return self._llm_chain.predict(topic=user_prompt, context=context)

class QuestionAnswer:
    def __init__(self, question: str, answer: str):
        self.question: str = question
        self.answer: str = answer
    
    def to_string(self) -> str:
        return f"Question: {self.question}, Answer: {self.answer}"

class Node:
    def __init__(self, topic: str, question_answers: List[QuestionAnswer] | None = None, children: List['Node'] | None = None):
        self.topic : str = topic
        self.question_answers : List[QuestionAnswer] = question_answers
        self.children : List['Node'] = children
    def to_string(self) -> str:
        result = "[ "
        for qa in self.question_answers:
            result += f" ( {self.topic}, {qa.to_string()} ), " 
        result += " ]"
        return result
    
def add_to_nodes(nodes: List[Node], topic: str, question_answer: QuestionAnswer):
    for node in nodes:
        if node.topic == topic:
            if node.question_answers is None:
                node.question_answers = []
            node.question_answers.append(question_answer)
            print (f"Added question {question_answer.to_string()} to topic | {topic} |")
            return
        
    print (f"Created new topic | {topic} | for question {question_answer.to_string()}")
    nodes.append(Node(topic, [question_answer]))

def get_all_categories(nodes: List[Node]) -> List[str]:
    categories : List[str] = []
    for node in nodes:
        categories.append(node.topic)
    return categories

def node_is_eligible_for_depth(node: Node) -> bool:
    if node.topic  != "Person" and node.question_answers is not None and len(node.question_answers) > 4:
        return True
    return False

def get_random_node(nodes: List[Node]) -> Node:
    
    if len(nodes) == 1:
        return nodes[0]

    nodes_eligible_for_depth = []
    nodes_eligible_for_breath = []

    for node in nodes:
        if node_is_eligible_for_depth(node):
            nodes_eligible_for_depth.append(node)
        else:
            nodes_eligible_for_breath.append(node)
    
    if len(nodes_eligible_for_depth) == 0:
        return nodes_eligible_for_breath[random.randint(0, len(nodes_eligible_for_breath) - 1)]

    breath : bool = random.choice([True, False])

    print(f"selected breath: {breath}")

    if breath:
        return nodes_eligible_for_breath[random.randint(0, len(nodes_eligible_for_breath) - 1)]
    
    return nodes_eligible_for_depth[random.randint(0, len(nodes_eligible_for_depth) - 1)]

def get_all_questionsAnswers(nodes: List[Node]) -> str|None:
    if len(nodes) == 0:
        return None
    result = "["
    for node in nodes:
        if node.question_answers is None:
            continue
        for qa in node.question_answers:
            result += f"( {node.topic}, {qa.to_string()} ),"
    result += "]"
    return result

def print_tree(nodes: List[Node]):
    print("\n====================== STATE OF TREE ==============================")
    for node in nodes:
        print(f"Topic: {node.topic}")
        if node.question_answers is None: 
            print("\tNo questions")
            continue
        for question_answer in node.question_answers:
            print(f"\tQuestion: {question_answer.question} Answer: {question_answer.answer}")
    print("====================================================================\n")

def init(profile_id: str):
    cache_filename = profile_id + ".pkl"
    connection_str = st.secrets["AZURE_STORAGE_CONNECTION_STRING"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_str)
    container_client = blob_service_client.get_container_client(container= container_name) 

    profile_list = container_client.list_blobs()
    for profile in profile_list:
        if profile.name == cache_filename:
            print(f"{cache_filename} Profile found in Azure Blob Storage. Downloading...")
            with open(cache_filename, "wb") as profile_file:
                download_stream = container_client.download_blob(profile.name)
                profile_file.write(download_stream.readall())
            break

def get_cached_nodes(profile_id: str) -> List[Node]:
    cache_filename = profile_id + ".pkl"
    nodes : List[Node] = []
    try:
        with open(cache_filename, 'rb') as in_file:
            nodes = pickle.load(in_file)
            print_tree(nodes)
    except FileNotFoundError:
        print(f"The cache file {cache_filename} does not exist.")
        nodes.append(Node("Person"))
    except ValueError:
        print("Unknown Error trying to read cache file.")
    return nodes

def cache_nodes(nodes: List[Node], cache_filename : str):
    write_nodes : List[Node] = []

    for node in nodes:
        all_qa : List[QuestionAnswer] = []
        for qan in node.question_answers:
            qa : QuestionAnswer = QuestionAnswer(qan.question, qan.answer)
            all_qa.append(qa)
        temp_node: Node = Node(node.topic, all_qa)
        write_nodes.append(temp_node)

    with open(cache_filename, 'wb') as out_file:
        pickle.dump(write_nodes, out_file)

    connection_str = st.secrets["AZURE_STORAGE_CONNECTION_STRING"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=cache_filename)
    print(f"Uploading to Azure Storage as blob:\t {cache_filename}")

    # Upload the created file
    with open(file=cache_filename, mode="rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    # Remove the local file
    os.remove(cache_filename)

def get_basic_questions() -> List[str]:
    basic_questions : List[str] = []
    basic_questions.append("What is your name?")
    basic_questions.append("How tall are you?")
    basic_questions.append("When is birthday?")
    basic_questions.append("Where do you live?")
    basic_questions.append("How do you identify your sexual orientation?")
    basic_questions.append("Who are you hoping to meet?")

    return basic_questions

def get_questions(nodes: List[Node]) -> List[str]:
    llm = AzureChatOpenAI(
    temperature=0.64,
    model= os.environ['GPT_4_32K_DEPLOYMENT'],
    request_timeout='60',
    max_retries='3',
    azure_deployment= os.environ['GPT_4_32K_DEPLOYMENT'],
    openai_api_version= os.environ['OPENAI_API_VERSION'],
    model_kwargs={}
)

    qa_depth_system_prompt = """Given known information on a person (in form of a list of topic, question and answer triplet) as context and a topic, come up with 3 relevant questions about the provided topic that one could ask to understand that area of a person's life more.
    Ask meaningful and in depth questions and use the provided context to come up with relevant questions. If no context is provided or topic is 'Person', generate 3 general questions that one could ask to get to know someone. The general questions can range from religion, politics, hobbies, career, childhood, dating preferences, etc..
    Respond with one question per line. Do not return any question that is similar to an existing one in the context.
    =========
    EXAMPLE:
    =========
    CONTEXT: [(Topic: childhood, Question: How many siblings do you have?, Answer: I have 2 siblings), (Topic: childhood, Question: What was your favorite activity as a child?, Answer: My favorite toy was playing soccer with my one of my siblings), (Topic: career, Question: What is your job?, Answer: I am a software engineer)]
    =========
    TOPIC: childhood
    =========
    FINAL ANSWER:
    Which one of our 2 siblings did you enjoy playing soccer with as a child?
    =========
    CONTEXT:
    {context}
    =========    
    TOPIC:
    {topic}
    =========
    """

    qa_breath_system_prompt = """Given known information on a person (in form of a list of topic, question and answer triplet) as context and a topic, come up with 3 questions about the provided topic that one could ask to understand that area of a person's life more.
    Do not return any question that is similar to an existing one in the context. Respond with one question per line. 
    =========
    EXAMPLE:
    =========
    CONTEXT: [(Topic: childhood, Question: How many siblings do you have?, Answer: I have 2 siblings), (Topic: childhood, Question: What was your favorite toy as a child?, Answer: My favorite toy was a teddy bear), (Topic: career, Question: What is your job?, Answer: I am a software engineer)]
    =========
    TOPIC: childhood
    =========
    FINAL ANSWER:
    What was your relationship with your parents like as child?
    =========
    CONTEXT:
    {context}
    =========    
    TOPIC:
    {topic}
    =========
    """

    qa_depth_agent = OpenAIAgent(llm, qa_depth_system_prompt)
    qa_breath_agent = OpenAIAgent(llm, qa_breath_system_prompt)

    unknown_person : bool = False

    basic_questions = get_basic_questions()

    context = "[]"
    
    current_node = get_random_node(nodes)
    depth_eligible = node_is_eligible_for_depth(current_node)
    topic = current_node.topic

    if current_node.topic == "Person" and current_node.question_answers is None:
        unknown_person = True

    if current_node.topic == "Person" and current_node.question_answers is not None:
        list_of_potential_topics = ["Childhood", "Career", "Hobbies", "Personality", "Religion", "Politics", "Dating Preferences"]
        topic = random.choice(list_of_potential_topics)

    temp_context = get_all_questionsAnswers(nodes)
    if temp_context is not None:
        context = temp_context

    print(f"Unknown Person? {unknown_person}")    
    print(f"The topic is {topic}")
    print(f"The context is {context}")
    print(f"Depth Eligible? {depth_eligible}")


    questions : List[str] | None  = basic_questions

    if not unknown_person:
        if depth_eligible:
            questions_in_str : str = qa_depth_agent.ask(topic, context)
        else:
            questions_in_str : str = qa_breath_agent.ask(topic, context)
        questions = questions_in_str.split('\n')

    result : List[str] = []
    for question in questions:

        if question is None or question == "" or question.isspace():
            continue
        result.append(question)
    
    return result

def process_response(nodes: List[Node], question: str, answer: str):
    question_answer = QuestionAnswer(question, answer)
    basic_questions = get_basic_questions()

    if question in basic_questions:
        add_to_nodes(nodes, "Person", question_answer)
        return

    llm = AzureChatOpenAI(
        temperature=0.64,
        model= os.environ['GPT_4_32K_DEPLOYMENT'],
        request_timeout='60',
        max_retries='3',
        azure_deployment= os.environ['GPT_4_32K_DEPLOYMENT'],
        openai_api_version= os.environ['OPENAI_API_VERSION'],
        model_kwargs={}
        )
    topic_system_prompt = """Given a pair of question and answer about a person as context and a list of categories as topic, deduce what category is the closest match for the question and answer pair.
    If there is not a close match or no topic is given, create and respond with a new category.
    Answer with just the topic name which can be one to a few words only.
    =========
    EXAMPLE:
    =========
    CONTEXT: Question: How many siblings do you have?, Answer: I have 2 siblings
    =========
    TOPIC: [Career, Hobbies, Personality]
    =========
    FINAL ANSWER:
    Childhood
    =========
    CONTEXT:
    {context}
    =========    
    TOPIC:
    {topic}
    =========
    """
    topic_agent = OpenAIAgent(llm, topic_system_prompt)
    category = topic_agent.ask(question_answer.to_string(), str(get_all_categories(nodes)))
    add_to_nodes(nodes, category, question_answer)

def get_summary(topic:str, nodes: List[Node]) -> str :
    llm = AzureChatOpenAI(
    temperature=0.64,
    model= os.environ['GPT_4_32K_DEPLOYMENT'],
    request_timeout='60',
    max_retries='3',
    azure_deployment= os.environ['GPT_4_32K_DEPLOYMENT'],
    openai_api_version= os.environ['OPENAI_API_VERSION'],
    model_kwargs={}
    )
    
    summary_system_prompt = """Given known information on a person (in form of a list of topic, question and answer triplet) as context and a topic, come up with a summary that describes the person around the topic in one paragraph.
    =========
    EXAMPLE:
    =========
    CONTEXT: [(Topic: childhood, Question: How many siblings do you have?, Answer: I have 2 siblings), (Topic: childhood, Question: What was your favorite toy as a child?, Answer: My favorite toy was a teddy bear), (Topic: career, Question: What is your job?, Answer: I am a software engineer, I always loved programming with computers since i was  kid)]
    =========
    TOPIC: childhood
    =========
    FINAL ANSWER:
    In terms of childhood, the person has 2 siblings and their favorite toy as a child was a teddy bear. They loved programming as a child as well.
    =========
    CONTEXT:
    {context}
    =========  
    TOPIC:
    {topic}
    =========  
    """
    summary_agent = OpenAIAgent(llm, summary_system_prompt)
    summary = summary_agent.ask(topic, get_all_questionsAnswers(nodes))
    print(f"\n {topic} : {summary} \n")

st.title("Welcome to TrueMatch")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "initizalized" not in st.session_state:
    st.session_state.initizalized = False

if "nodes" not in st.session_state:
    st.session_state.nodes = []

if "need_new_questions" not in st.session_state:
    st.session_state.need_new_questions = True

if "waiting_for_answer" not in st.session_state:
    st.session_state.waiting_for_answer = False

if "question_index" not in st.session_state:
    st.session_state.question_index = 0

if "questions" not in st.session_state:
    st.session_state.questions = []

if "question" not in st.session_state:
    st.session_state.question = ""

if "option_to_end_session" not in st.session_state:
    st.session_state.option_to_end_session = False

if "cont" not in st.session_state:
    st.session_state.cont = True

if "session_complete" not in st.session_state:
    st.session_state.session_complete = False

if "asked_for_id" not in st.session_state:
    st.session_state.asked_for_id = False

if "asked_for_resume" not in st.session_state:
    st.session_state.asked_for_resume = False

if "user_id" not in st.session_state:
    st.session_state.user_id = ""

if "authorized" not in st.session_state:
    st.session_state.authorized = True

if not st.session_state.initizalized:
    if not st.session_state.asked_for_id:
        response = "Hello, please provide your user id to get started"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.asked_for_id = True

    if user_id := st.chat_input("Your user_id here"):
        allowed_users = connection_str = st.secrets["ALLOWED_USERS"]
        if user_id not in allowed_users:
            st.session_state.authorized = False
            st.session_state.session_complete = True
            st.session_state.initizalized = True
            st.session_state.cont = False
            st.rerun()
        st.session_state.user_id = user_id
        st.session_state.messages.append({"role": "user", "content": user_id})
        
        with st.chat_message("user"):
            st.markdown(user_id)
            init(user_id)
            st.session_state.nodes = get_cached_nodes(user_id)
            st.session_state.initizalized = True

        response = "Looking you up now.... "
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if st.session_state.initizalized and st.session_state.need_new_questions and st.session_state.cont and not st.session_state.session_complete:
    st.session_state.questions = get_questions(st.session_state.nodes)
    st.session_state.need_new_questions = False

if not st.session_state.need_new_questions and not st.session_state.waiting_for_answer and st.session_state.cont and not st.session_state.session_complete:
    st.session_state.question = st.session_state.questions[st.session_state.question_index]
    with st.chat_message("assistant"):
        st.markdown(st.session_state.question)
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.question})
    st.session_state.waiting_for_answer = True
    st.rerun()

if st.session_state.waiting_for_answer and st.session_state.cont and not st.session_state.session_complete:
    if answer := st.chat_input("Your answer to the question"):
        st.session_state.messages.append({"role": "user", "content": answer})
        process_response(st.session_state.nodes, st.session_state.question, answer)

        with st.chat_message("user"):
            st.markdown(answer)    

        st.session_state.question_index = st.session_state.question_index + 1
        if st.session_state.question_index == len(st.session_state.questions):
            st.session_state.option_to_end_session = True
            st.session_state.need_new_questions = True
            st.session_state.question_index = 0
            st.session_state.cont = False
        st.session_state.waiting_for_answer = False 
        st.rerun()

if st.session_state.option_to_end_session and not st.session_state.session_complete:
    st.session_state.cont = False

    if not st.session_state.asked_for_resume:
        message = "That's yet one batch of questions. You have the option to continue or save your progress and continue later. Would you like to continue?"
        with st.chat_message("assistant"):
            st.markdown(message)
        st.session_state.messages.append({"role": "assistant", "content": message})
        st.session_state.asked_for_resume = True

    if response := st.chat_input("Response"):
        st.session_state.messages.append({"role": "user", "content": response})
        
        with st.chat_message("user"):
            st.markdown(response)
            if response == 'No' or response == 'no' or response == 'NO':
                cache_nodes(st.session_state.nodes, st.session_state.user_id + '.pkl')
                st.session_state.session_complete = True
            else:
                st.session_state.cont = True
                st.session_state.option_to_end_session = False
                st.session_state.asked_for_resume = False
            st.rerun()

if st.session_state.session_complete:
    if not st.session_state.authorized:
        with st.chat_message("assistant"):
            st.markdown("You are not authorized to access this pilot application yet.")
    message = "Thank you for visiting us today. Hope to connect again soon."
    with st.chat_message("assistant"):
        st.markdown(message)
    st.session_state.messages.append({"role": "assistant", "content": message})
    st.stop()
