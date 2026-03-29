import os
import json
import re
import logging
import traceback
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

try:
    import openai
except ImportError:
    openai = None

from embodied_eval.evaluators.eqa_evaluator import EQAEvaluator


def extract_number(text):
    """Using regular expression to find number in text"""
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def is_binary_question(gts):
    """Check if this is a binary (yes/no) question"""
    for gt in gts:
        if 'yes' in gt.lower() or 'no' in gt.lower():
            return 1
    return 0


def call_openai_api_azure(
    messages: list,
    api_key: str = None,
    model: str = 'gpt-4o-2024-08-06',
    region: str = 'northcentralus',
):
    """Call OpenAI Azure API"""
    if openai is None:
        raise ImportError("openai package is required for LLM evaluation")

    API_BASE = ""   # TODO: Fill in your Azure API base URL
    ENDPOINT = f"{API_BASE}/{region}"

    if api_key is None:
        if 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        elif 'AZURE_OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
        else:
            raise LookupError("No OpenAI API key found. Please set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.")

    client = openai.AzureOpenAI(
        api_key=api_key,
        api_version='2024-02-01',
        azure_endpoint=ENDPOINT,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        traceback.print_exc()
        raise e


def call_openai_api(
    messages: list,
    api_key: str = None,
    model: str = 'gpt-4o-2024-08-06',
):
    """Call OpenAI API directly"""
    if openai is None:
        raise ImportError("openai package is required for LLM evaluation")

    if api_key is None:
        if 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        elif 'AZURE_OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
        else:
            raise LookupError("No OpenAI API key found. Please set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.")

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        traceback.print_exc()
        raise e


def clean_answer(data):
    """Clean and normalize answer text (from SQA3D)"""
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


def answer_match(pred, gts):
    """Return EM and refined EM"""
    for gt in gts:
        if pred == gt:
            return 1, 1
        elif ''.join(pred.split()) in ''.join(gt.split()):
            return 0, 1
        elif ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0


class OfficialBeacon3DLLEvaluator():
    """Official Beacon3D LLM Evaluator using GPT-4 for scoring"""

    def __init__(self, model='gpt-4o-2024-08-06', region='northcentralus',
                 prompt_path='/your/path/to/embodied-eval-main/embodied_eval/data/Beacon3D/data/system_prompt.json',
                 use_azure=True, verbose=False):
        self.model = model
        self.region = region
        self.use_azure = use_azure
        self.verbose = verbose

        # Load system prompt
        try:
            with open(prompt_path) as f:
                self.messages = json.load(f)
        except FileNotFoundError:
            # Fallback to default prompt
            self.messages = [{
                "role": "system",
                "content": "Score open-ended answers from 1 to 5 based on accuracy, completeness, and relevance to the ground truth.\nCriteria:\nCounting:\nQuestion: How many tables are in the room?\nGround Truth: Three\nExample Response: Two\nScore: 1 (Significant discrepancy)\nExistence:\nQuestion: Is there a chair on my left?\nGround Truth: Yes\nExample Response: Yes, there is a chair on the left.\nScore: 5 (Correct match)\nDescription:\nQuestion: Describe the <couch-3-IMG> on my left.\nGround Truth: Black couch with yellow and orange pillows.\nExample Response: Multicolored bed.\nScore: 1 (Incorrect identification)\nSpatial Relationship:\nQuestion: What is the relationship between the desk and the computer tower?\nGround Truth: Inside the desk.\nExample Response: To the right of the desk.\nScore: 1 (Incorrect relationship)\nQuestion: What is the relationship between the chair and the table?\nGround Truth: On the left of the table.\nExample Response: On the left of the table.\nScore: 5 (Correct match)\nNavigation:\nQuestion: How do I reach the window on my right from my current position?\nGround Truth: Turn right and walk to the middle distance.\nExample Response: Turn to your right.\nScore: 3 (Partial instructions)\nQuestion: Where is the wooden chair located in relation to me?\nGround Truth: At your 6 o'clock.\nExample Response: At your 7 o'clock.\nScore: 4 (Minor discrepancy)\nExample Response: At your 10 o'clock.\nScore: 1 (Major discrepancy)\nObject Reference:\nQuestion: What is on the table behind me?\nGround Truth: A book, a pencil, and a monitor.\nExample Response: a pencil and a monitor.\nScore: 3 (Partial context)\nRoom Type or Affordance: (Provide context-specific details and examples)\nGuidelines:\nScore 5: Perfect or highly accurate response.\nScore 1: Significant inaccuracies or discrepancies.\nScore 2-4: Reflect partial correctness or minor errors.\nOutput only the score.\n"
            }]

    def score(self, question, answer, gt):
        """Score a single QA pair using LLM"""
        messages = self.messages.copy()
        user_prompt = '\n'.join([f"Question: {question}", f"Answer: {answer}", f"Ground Truth: {gt}"])
        messages.append({'role': 'user', 'content': user_prompt})

        if self.use_azure:
            response = call_openai_api_azure(messages=messages, model=self.model, region=self.region)
        else:
            response = call_openai_api(messages=messages, model=self.model)

        score = extract_number(response)
        if self.verbose:
            print(user_prompt, score)
        return score


class OfficialBeacon3DEvaluator(EQAEvaluator):
    """Official Beacon3D Evaluator that matches the official evaluation exactly"""

    def __init__(self, args):
        super().__init__(args)

        # Initialize LLM evaluator for Beacon3D tasks
        self.llm_evaluators = {}

        # Check if we have Beacon3D tasks
        beacon3d_tasks = [task for task in self.tasks if 'beacon3d' in task.lower()]
        if beacon3d_tasks:
            try:
                self.llm_evaluators['beacon3d'] = OfficialBeacon3DLLEvaluator(
                    model='gpt-4o-2024-08-06',
                    region='northcentralus',
                    prompt_path='/your/path/to/embodied-eval-main/embodied_eval/data/Beacon3D/data/system_prompt.json',
                    use_azure=True,
                    verbose=False
                )
                print("✅ Official Beacon3D LLM evaluator initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize LLM evaluator: {e}")
                print("Falling back to basic evaluation")
                self.llm_evaluators['beacon3d'] = None

    def evaluate(self, eval_tasks):
        """Override evaluate to use official Beacon3D scoring"""
        results_dict = {}
        results = {}
        configs = {}
        samples = {}

        RANK = self.model.rank
        WORLD_SIZE = self.model.world_size

        for task in eval_tasks:
            if RANK == 0:
                print(f"{'-'*50} Evaluating {task} {'-'*50}")

            config = self.model.task_dict[task]
            test_set = self.model.task_dict[task][self.split]
            replicates = self.limit if self.limit else len(test_set)

            if self.limit:
                np.random.seed(42)
                test_set = np.random.choice(test_set, size=self.limit, replace=False)

            # Process results using official evaluation
            if 'beacon3d' in task.lower() and self.llm_evaluators.get('beacon3d'):
                processed_results = self._evaluate_beacon3d_official(task, eval_tasks[task])
            else:
                # Fall back to standard evaluation
                processed_results = self._evaluate_standard(eval_tasks[task])

            # Store results
            results[task] = processed_results
            configs[task] = config
            samples[task] = eval_tasks[task]

        # Save results and compute final metrics
        if RANK == 0:
            for task in eval_tasks:
                if 'beacon3d' in task.lower() and self.llm_evaluators.get('beacon3d'):
                    self._save_and_print_beacon3d_results(task, results[task], samples[task])
                else:
                    self.print_results({task: results[task]})

                if self.output_path:
                    os.makedirs(self.output_path, exist_ok=True)
                    with open(os.path.join(self.output_path, f"results_{task}.json"), 'w') as f:
                        json.dump(results[task], f, indent=2)

        return results

    def _evaluate_beacon3d_official(self, task_name, inference_results):
        """Evaluate Beacon3D using official LLM-based scoring"""
        print("Using official Beacon3D LLM evaluation...")

        # Load test data and metadata
        data_path = '/data/arena/datasets/Beacon3D/data/scannet/scannet_qa.json'
        metadata_path = '/data/arena/datasets/Beacon3D/data/scannet/metadata_scannet_qa.json'

        with open(data_path) as f:
            data = json.load(f)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Extract predictions from inference results
        predictions = {}
        for item in inference_results:
            doc_id = item.get('doc_id')
            pred = self._extract_prediction(item)
            if pred is not None:
                predictions[doc_id] = pred

        # Build results mapping (following official evaluation)
        infer_results_mapping = {}
        for item1 in tqdm(data, desc="Processing predictions"):
            scene_id = item1['scene_id']
            question = item1['question']
            gt = item1['answers']

            doc_id = item1.get('doc_id') or item1.get('question_id')
            pred = predictions.get(doc_id, "")

            pred = clean_answer(pred) if pred else ""
            gt = [clean_answer(_gt) for _gt in gt]

            em, em_r = answer_match(pred, gt)

            if em:
                score = 5
            elif is_binary_question(gt) and em_r:
                score = 5
            else:
                try:
                    score = self.llm_evaluators['beacon3d'].score(question, pred, gt[0])
                except Exception as e:
                    print(f"LLM scoring failed: {e}, using default score 1")
                    score = 1

            if scene_id not in infer_results_mapping:
                infer_results_mapping[scene_id] = {}

            tag, extra = item1['question_id'].split('_')[-2:]
            infer_results_mapping[scene_id][question.lower()] = (pred, em, em_r, score, tag, int(extra))

        # Process to metadata format
        processed_qa = self._process_to_metadata(metadata, infer_results_mapping)

        return processed_qa

    def _extract_prediction(self, item):
        """Extract prediction from inference result"""
        if 'resps' in item and item['resps']:
            if isinstance(item['resps'][0], list):
                return item['resps'][0][0] if item['resps'][0] else ""
            return item['resps'][0]
        elif 'prediction' in item:
            return item['prediction']
        elif 'text' in item:
            return item['text']
        return ""

    def _process_to_metadata(self, metadata, infer_results_mapping):
        """Process inference results according to metadata format"""
        output = []
        for scene_id, v0 in metadata.items():
            for obj_class, v1 in v0.items():
                for obj_id, v2 in v1.items():
                    for item in v2['qa']:
                        q = item['question'].lower()
                        tag = item['tag']
                        extra = int(item['extra_knowledge'])
                        assert tag == infer_results_mapping[scene_id][q][4] and extra == infer_results_mapping[scene_id][q][5]
                        this_dict = {
                            'obj_id': f'{scene_id}-{obj_id}',
                            'question': q,
                            'tag': tag,
                            'extra_knowledge': extra,
                            'answer_gt': item['answer'],
                            'answer_pred': infer_results_mapping[scene_id][q][0],
                            'em': infer_results_mapping[scene_id][q][1],
                            'em_r': infer_results_mapping[scene_id][q][2],
                            'score': infer_results_mapping[scene_id][q][3],
                        }
                        output.append(this_dict)
        return output

    def _evaluate_standard(self, inference_results):
        """Fallback to standard evaluation"""
        return super().evaluate({'temp': inference_results})['temp']

    def _save_and_print_beacon3d_results(self, task_name, processed_qa, samples):
        """Print results in official Beacon3D format"""
        print(f"\n{'='*60}")
        print(f"Official Beacon3D {task_name.upper()} Results")
        print(f"{'='*60}")

        # Overall metrics
        overall_em = np.mean([item['em'] for item in processed_qa]) * 100
        overall_em_r = np.mean([item['em_r'] for item in processed_qa]) * 100
        overall_gpt_score = np.mean([item['score'] for item in processed_qa])
        overall_gpt_score = (overall_gpt_score - 1) / 4 * 100

        print(f"EM = {overall_em:.2f} | EM-R = {overall_em_r:.2f} | GPT-Score = {overall_gpt_score:.2f}")

        # Statistics
        print("\nData statistics:")
        total = len(processed_qa)
        print(f"case: {total}")
        print(f"object: {len(range(0, total, 3))}")

        # Tag statistics
        knowledge_types = ['class', 'appearance', 'geometry', 'spatial', 'existence', 'functionality']
        tag_count_wo_extra = {}
        tag_count_w_extra = {}
        count_wo_extra = 0
        count_w_extra = 0

        for item in processed_qa:
            tag = item['tag'].lower()
            extra = item['extra_knowledge']
            if extra:
                if tag not in tag_count_w_extra:
                    tag_count_w_extra[tag] = 0
                tag_count_w_extra[tag] += 1
                count_w_extra += 1
            else:
                if tag not in tag_count_wo_extra:
                    tag_count_wo_extra[tag] = 0
                tag_count_wo_extra[tag] += 1
                count_wo_extra += 1

        print(f"[w/o extra]: {count_wo_extra}")
        print(f"[w/ extra]: {count_w_extra}")

        # Scores
        print("\nScores:")
        case_scores = [item['score'] for item in processed_qa]
        case_metrics = (np.mean(case_scores)- 1) / 4 * 100
        print(f"case: {case_metrics:.2f}")

        # Per object metrics
        obj_scores = [case_scores[i:i+3] for i in range(0, total, 3)]
        obj_metrics = []
        for obj_score in obj_scores:
            if all([score >= 4 for score in obj_score]):
                obj_metrics.append(1)
            else:
                obj_metrics.append(0)
        obj_metrics = np.mean(obj_metrics) * 100
        print(f"object: {obj_metrics:.2f}")

        # Per tag metrics
        tag_metrics = {}
        for item in processed_qa:
            tag = item['tag'].lower()
            score = item['score']
            if tag not in tag_metrics:
                tag_metrics[tag] = []
            tag_metrics[tag].append(score)

        for k in knowledge_types:
            if k in tag_metrics:
                v = tag_metrics[k]
                print(f"{k}: {(np.mean(v) - 1) / 4 * 100:.2f}")

        print(f"{'='*60}\n")
