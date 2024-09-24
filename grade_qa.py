import glob
import json
import random
import os
import torch  # Import torch to use torch.bfloat16
from vllm import LLM, SamplingParams
sampling_params = SamplingParams(temperature=0, max_tokens=7)
llm = LLM(model="meta-llama/Meta-Llama-3.1-8B", dtype=torch.float16)

# Example QA pairs with grades and critiques
examples = [
    """Question: What were the derived sub-solar elemental abundance ratios for Complex C?
Answer: The derived sub-solar elemental abundance ratios for Complex C were [Fe/S] = -0.42 ± 0.08, [Si/S] = -0.29 ± 0.05, and [Al/S] = -0.53 ± 0.08.
Grades: [5,3,5,3]
Critique: derived where? by whom?""",
    """Question: How does the LOFAR LBA Sky Survey contribute to the study of cosmic rays and their interactions in galaxy clusters?
Answer: The LOFAR LBA Sky Survey will enable the investigation of cosmic-ray acceleration processes in both radio haloes and relics, contributing to understanding how cosmic rays interact within the intra-cluster medium.
Grades: [5,5,5,5]
Critique: N/A""",
    """Question: What unique features of ice giants like Uranus and Neptune are highlighted in the NUIP Mission, and how do they differ from gas giants?
Answer: Ice giants are primarily composed of heavier elements and have smaller hydrogen envelopes compared to gas giants, which are over 90% hydrogen and helium by mass. This fundamental difference influences their internal structures and atmospheric dynamics.
Grades: [5,5,5,5]
Critique: N/A""",
    """Question: What implications do the results have for the TGF detection rate from the AGILE satellite?
Answer: The results suggest a near doubling of the AGILE TGF detection rate, indicating improved identification of TGFs correlated with lightning events.
Grades: [5,1,5,4]
Critique: results of what?""",
    """Question: What is introduced in the model to account for the feedback from the first Population III stars before the formation of Population II stars?
Answer: A delay time of 10 Myr is introduced between PopIII and PopII star formation to account for the feedback from the first stars.
Grades: [5,1,3,3]
Critique: what model? why would someone care?""",
    """Question: What is the significance of the thermal conductance and heat capacity analysis in understanding MKIDs?
Answer: The thermal conductance and heat capacity analysis reveal the contributions from electrons, phonons, and TLS, highlighting the importance of TLS at low temperatures in influencing the overall thermal dynamics of the MKID.
Grades: [5,3,5,5]
Critique: which analysis?""",
    """Question: What does the hierarchical nature of star formation in the studied galaxies indicate about star-forming structures?
Answer: The hierarchical nature indicates that star formation occurs in localized clumps with varying extents and characteristics.
Grades: [5,3,5,4]
Critique: galaxies that were studied where?""",
    """Question: What are the essential requirements for sustainable human presence on Mars?
Answer: Sustainable human presence on Mars requires in-situ resource utilization technologies for extracting water, producing oxygen, and utilizing Martian regolith for construction. These technologies are critical for supporting life and ensuring long-term habitation.
Grades: [5,5,5,5]
Critique: N/A""",
    """Question: What does the investigation into the formation of \\\\beta Pic b suggest about its metallicity and C/O ratio?
Answer: \\\\beta Pic b is suggested to have a supersolar metallicity and a subsolar C/O ratio.
Grades: [5,2,2,2]
Critique: investigation by whom? leading question.""",
    """Question: Why is it challenging to apply mass-dependent clustering methods to observational data?
Answer: Applying mass-dependent clustering methods to observational data is challenging due to the difficulties in accurately measuring halo masses. Precise mass measurements are crucial for effectively reducing stochasticity and optimizing mass reconstruction.
Grades: [5,3,5,4]
Critique: clustering of what?""",
    """Question: What was the area covered by the COSMOS-Web treasury program, and what imaging filters were used?
Answer: The COSMOS-Web treasury program covered an area of 0.54 deg² using NIRCam imaging in four filters: F115W, F150W, F277W, and F444W.
Grades: [5,5,5,5]
Critique: N/A""",
]


def build_prompt(qa_pair):
    # Select two examples at random
    selected_examples = random.sample(examples, 2)
    prompt = f"""Grade the following question-answer pair based on these criteria:
1. Correctness: The factual accuracy of the answer provided. Does it answer the question and align with current scientific understanding?
2. Stand-alone: The question should be understandable without external context. Could an expert answer without requiring further context?
3. Pertinence: The question's importance to someone studying astronomy, astrophysics, or cosmology. Would this be a question a real person might ask?
4. Overall: Your overall impression of the question-answer pair. Does it help teach or inform about the topic?

Score each criterion from 1-5, where 1 is poor and 5 is excellent. Provide the scores in the format: [Correctness, Stand-alone, Pertinence, Overall]. For example, [5, 4, 3, 5] would indicate a score of 5 for Correctness, 4 for Stand-alone, 3 for Pertinence, and 5 Overall.

{selected_examples[0]}

{selected_examples[1]}

Q: {qa_pair['question']}
A: {qa_pair['answer']}
Grades: ["""
    return prompt


def parse_response(response):
    response = response.strip()
    try:
        # Extract the grades from the response
        grades = response.strip("[]").split(",")
        grades = [grade.strip() for grade in grades]
        assert len(grades) == 4
        grades = [int(grade) for grade in grades]
        assert all(1 <= grade <= 5 for grade in grades)
        total_grade = sum(grades)
        return total_grade
    except (AssertionError, ValueError, IndexError):
        print(f"Error parsing grade: {response}")
        return -1


# Define the maximum number of papers to process in a batch
max_papers_per_batch = 100

# Get the list of QA files
qa_raw_files = glob.glob("datasets/astrosage_qa/*.json")
qa_raw_files.sort()  # Optional: sort the files

total_files = len(qa_raw_files)

while True:
    # Prepare data structures for the batch
    prompts = []
    qa_pairs_list = []
    qa_pair_indices = []
    papers = []
    num_papers_processed = 0

    for qa_raw_file in qa_raw_files:
        output_file = os.path.join(
            "datasets/astrosage_qa/graded/", os.path.basename(qa_raw_file)
        )
        lock_file = output_file + ".lock"

        if os.path.exists(lock_file) or os.path.exists(output_file):
            print(f"Skipping {qa_raw_file} because lock file or output file exists.")
            continue  # Skip this paper

        if num_papers_processed >= max_papers_per_batch:
            break  # Process the batch

        # Create lock file
        with open(lock_file, "w") as f:
            f.write("Processing")

        # Load the paper
        with open(qa_raw_file, "r") as f:
            paper = json.load(f)
            print(
                f"Processing {paper['shorthand_title']} with {len(paper['qa_pairs'])} QA pairs."
            )

        # Keep track of the paper and its QA pairs
        papers.append(
            {"file_path": qa_raw_file, "paper": paper, "qa_pairs": paper["qa_pairs"]}
        )

        num_papers_processed += 1

        # Build prompts and keep track of indices
        for idx, qa_pair in enumerate(paper["qa_pairs"]):
            if not isinstance(qa_pair, dict):
                print(f"Skipping QA pair at index {idx} because it's not a dictionary.")
                continue            
            try: 
                prompt = build_prompt(qa_pair)
            except Exception as e:
                print(f"Error building prompt for QA pair.")
                print(e)
                continue
            prompts.append(prompt)
            qa_pairs_list.append(qa_pair)
            qa_pair_indices.append(
                (len(papers) - 1, idx)
            )  # Index to locate the QA pair

    if not prompts:
        print("No new prompts to process.")
        break  # All files are processed, break out of the while loop

    # Process the prompts
    print(
        f"Processing batch of {len(prompts)} prompts from {num_papers_processed} papers."
    )
    outputs = llm.generate(prompts, sampling_params)

    # Parse and assign grades
    for i, output in enumerate(outputs):
        response = output.outputs[0].text.strip()
        grade = parse_response(response)
        paper_idx, qa_idx = qa_pair_indices[i]
        qa_pair = papers[paper_idx]["qa_pairs"][qa_idx]
        qa_pair["grade"] = grade
        print(f"Graded QA pair with grade {grade}: {qa_pair['question']}")

    # Now, write out the papers and delete lock files
    for paper_info in papers:
        paper = paper_info["paper"]
        file_path = paper_info["file_path"]
        output_file = os.path.join(
            "datasets/astrosage_qa/graded/", os.path.basename(file_path)
        )
        lock_file = output_file + ".lock"

        # Ensure the graded directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(paper, f, indent=2)
            print(f"Saved graded file to {output_file}")

        # Delete the lock file
        if os.path.exists(lock_file):
            os.remove(lock_file)