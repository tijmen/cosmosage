"""
Add QA pairs to arxiv paper pkl file.
"""
import glob
import random
import pickle
import argparse
import logging
import multiprocessing
import re
from multiprocessing import Pool
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate QA from arXiv papers.")
parser.add_argument(
    "--port", type=int, help="Port number for the VLLM server", default=8000
)
args = parser.parse_args()
vllm_port = args.port

# filenames = glob.glob("input/datasets/arxiv_cache/*.pkl")
filenames = glob.glob("/home/tijmen/cosmosage/datasets/arxiv_cache/*.pkl")
random.shuffle(filenames)

logging.info(f"Found {len(filenames)} files to process.")


def generate_qa_pair(args):
    text = args["text"]
    summary = args["summary"]
    arxiv_id = args["arxiv_id"]
    title = args["title"]
    shorthand_title = args["shorthand_title"]

    class question_answer(BaseModel):
        question: str = Field(..., description="Question framed.")
        answer: str = Field(..., description="Answer to the question.")

    class output(BaseModel):
        output: list[question_answer] = []

    parser = PydanticOutputParser(pydantic_object=output)

    system_prompt = f"""Assume the role of a cosmology lecturer with a deep understanding of all areas of modern cosmology, astrophysics. You must precisely follow the instructions given by the user."""

    user_prompt = f"""As a cosmology expert, your task is to create precise and self-contained question-answer pairs from a specified PASSAGE of {shorthand_title}.
The paper is titled {title} and has the arXiv ID {arxiv_id}. Here is a short summary:
{summary}
Ensure that each question incorporates all necessary context, allowing it to be fully understood on its own. 
Answers should be clear, specific, and provide comprehensive information based on the PASSAGE. 
The goal is for each question and answer pair to be understandable independently, ensuring they are complete and contextually clear without external references.
"""

    # Additional instructions to add some variety
    bonus_instruction = [
        "Questions should probe different aspects of the content, encouraging a variety of answers.",
        "Formulate questions that challenge or dissect key points, theories, or data.",
        "Focus on comparing and contrasting ideas or findings with other known theories or data in cosmology.",
        "Create questions based on hypothetical scenarios or 'what-if' questions inspired by the PASSAGE.",
        "Form questions and answers focusing on the practical applications and implications of the research findings."
        "Formulate questions about potential future research directions or unanswered questions that arise from the study's findings.",
        "Delve into the technical aspects or methodologies used in the study. Ask questions that require detailed explanations of the processes, techniques, or calculations presented.",
        "Create questions that explore connections between the study's findings and other scientific disciplines, such as physics, mathematics, or computer science.",
        "Ask questions that consider the broader philosophical implications or ethical considerations of the research findings in the field of cosmology.",
        "Don't forget to have fun!",
        "Remember to be creative!",
        "Feel free to reply with a sense of humor.",
        "Be sure to include all relevant details in your answers.",
        "Ensure that each question is clear and understandable on its own.",
        "Make sure that each answer is clear and specific.",
        "Ensure that each answer is comprehensive and complete.",
        "Make sure that each question and answer pair is understandable independently.",
        "Ensure that each question and answer pair is contextually clear.",
        "Ensure that each question and answer pair is complete.",
        "Be edgy, harsh and critical. Off-the wall is ok. I like bonkers!",
        "Consider whether a mathematical representation would be helpful in summarizing the PASSAGE.",
    ]

    user_prompt += random.choice(bonus_instruction)

    user_prompt += """

PASSAGE: {text}

Format instructions: {format_instructions}"""

#     prompt = f"""<|im_start|>system
# {system_prompt}<|im_end|>
# <|im_start|>user
# {user_prompt}<|im_end|>
# <|im_start|>assistant"""
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=PromptTemplate(template=user_prompt, input_variables=["text"], partial_variables={"format_instructions": parser.get_format_instructions()}).format_prompt(text=text).to_string())]

    # connect to the VLLM server that I started separately with something like
    logging.info(f"Connecting to VLLM server at port {vllm_port}")
    inference_server_url = f"http://localhost:{vllm_port}/v1"

    llm = ChatOpenAI(
        model="/home/tijmen/public_models/TheBloke_Nous-Hermes-2-Yi-34B-GPTQ_gptq-4bit-32g-actorder_True",
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
        max_tokens=1800,
        temperature=0.4,
    )

    llm_response = llm(messages).content

    # Check if the response is not empty or None
    if not llm_response:
        logging.warning("The response from the LLM is empty or None.")
        return []

    def extract_qa_pairs(text):
        qa_pairs = []

        def match_patterns(text):
            # Multiple patterns to account for different structures
            patterns = [
                r'\{\s*"question":\s*"(.*?)"\s*,\s*"answer":\s*"(.*?)"\s*\}',  # Original pattern
                r"\"question\":\s*\"(.*?)\"\s*,\s*\"answer\":\s*\"(.*?)\"",  # Pattern for nested within an array
                r'[{[]\s*\\?"question\\?":\s*\\?"(.*?)\\?"\s*,\s*\\?"answer\\?":\s*\\?"(.*?)\\?"\s*[}\]]',  # Pattern with escaped quotes
                r'\\?"question\\?":\s*\\?"(.*?)\\?"\s*,\s*\\?"answer\\?":\s*\\?"(.*?)\\?"(?=,\s*\\?[{[]|\s*\\?]\])',  # Pattern for multiple JSON objects in an array
                r'[{[]\s*\\?"output\\?":\s*\[\s*{.*?"question":\s*\'(.*?)\'\s*,\s*"answer":\s*\'(.*?)\'\s*}(?:,\s*{.*?}|])',  # Pattern for nested structure with single quotes
                r'"question":\s*"(.+?)"\s*,\s*"answer":\s*"((?:[^"]|"(?![},]))+)"',  # New pattern to capture multi-sentence answers
                r"\"question\":\s*\"(.*?)\"\s*,\s*\"answer\":\s*\{(.*?)\}(?=\s*,|\s*\])",  # Pattern to capture nested answer object
                r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\s*Question:|\s*$)", # pattern for plaintext questions and answers
                r"\d+\.\s*(.*?)\s*Answer:\s*(.*?)(?=\s*\d+\.|\s*$)"  # New pattern for numbered QA pairs
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    return matches
            # If no pattern matches
            logging.warning(f"No matches found for patterns in text: {text}")
            return None

        matches = match_patterns(text)
        if not matches:
            return []

        for question, answer in matches:
            try:
                # Manually construct the dictionary from question and answer
                qa_pair = {
                    "question": question.replace("\n", " ").replace('\\"', '"'),
                    "answer": answer.replace("\n", " ").replace('\\"', '"'),
                }
                qa_pairs.append(qa_pair)
            except Exception as e:
                logging.warning(f"Error constructing QA pair: {e}")
                logging.warning("Failed match:", question, answer)

        return qa_pairs

    output_list = extract_qa_pairs(llm_response)
    return output_list


def chunk_text(text, max_chunk_size=1800, overlap=600):
    """
    Divide text into overlapping chunks, attempting to end each chunk at sentence boundaries.
    """
    sentences = text.split(".")
    chunks = []
    current_chunk = ""
    overlap_buffer = ""

    for sentence in sentences:
        # Add back the period removed by split, except for the last sentence
        sentence = (
            sentence.strip() + "." if sentence != sentences[-1] else sentence.strip()
        )
        sentence += " "  # Add space after each sentence for readability

        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            # Store the current chunk and start a new one
            chunks.append(current_chunk)
            # Start new chunk with the end of the previous chunk (for overlap)
            current_chunk = overlap_buffer + sentence
            overlap_buffer = ""  # Clear the overlap buffer

        # Update overlap buffer
        overlap_words = sentence.split()[
            -overlap:
        ]  # Get last 'overlap' words from sentence
        overlap_buffer = " ".join(overlap_words) + " "  # Add space for readability

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def add_QA(filename):
    with open(filename, "rb") as f:
        paper = pickle.load(f)
    if "QA" in paper:
        logging.info(f"QA already exists for {filename}. Skipping...")
        return
    logging.info(f"Generating QA for {filename}...")
    try:
        qa_pairs = []

        for chunk in chunk_text(paper["text"]):
            args = {
                "text": chunk,
                "summary": paper["summary"],
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "shorthand_title": paper["shorthand_title"],
            }
            
            try:
                qa_pairs.extend(generate_qa_pair(args))
            except Exception as e:
                logging.warning(f"Failed to generate this QA pair: {e}")

        # replace any occurences of "the paper" or "the study" with the shorthand title.
        for qa_pair in qa_pairs:
            for word in [
                "the paper",
                "this paper",
                "the study",
                "this study",
                "this research",
            ]:
                qa_pair["question"] = qa_pair["question"].replace(
                    word, paper["shorthand_title"]
                )
                qa_pair["answer"] = qa_pair["answer"].replace(
                    word, paper["shorthand_title"]
                )

        paper["qa"] = qa_pairs

    except Exception as e:
        logging.warning(f"!*_*! Failure in generating QA pairs for {filename}: {e} !*_*!")
        return

    with open(filename, "wb") as f:
        pickle.dump(paper, f)
        logging.info(f"Succesfully added synthetic QA data to {filename}!")

# # Create a pool of workers
# logging.info("Creating pool of workers")
# pool = multiprocessing.Pool(16)

# # Map the process_arxiv_id function to each arxiv_id in parallel
# logging.info("Mapping add_QA to filenames")
# for filename in filenames:
#     pool.apply_async(add_QA, args=(filename,))

# # Close the pool of workers
# logging.info("Closing pool of workers")
# pool.close()
# logging.info("Waiting for workers to finish")
# pool.join()


add_QA(filenames[0])
