import os
import shutil

import numpy as np
import spacy
import pyrallis


from src.similarity_score_calculation import text2image_noun_score, text2image_rel_score, image2text_score
from src.matching import flip_matching
from src.making_questions import parse_output
from src.llm_api import LllmApi
from src.utils import read_txt_file, read_prompt_question_from_api, read_meta_data_question_from_api
from src.config import RunConfig
from src.vqa import VQAModel
from yolov9.convert_to_our_json import convert_to_dictionary
from yolov9.making_boxes import get_relation_boxes_in_json, handle_no_detection
from yolov9.detect import run


def object_detection(image_dir):
    run(
        weights='yolov9/yolov9-e-converted.pt',
        source=image_dir,
        data='yolov9/data/coco.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='yolov9/runs/detect',  # save results to project/name
        name='Diffusion_Generation',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,
        dnn=False,
        vid_stride=1,)

    noun_objects_dict = dict(convert_to_dictionary(
        path_to_directory="yolov9/runs/detect/Diffusion_Generation/labels"))
    noun_objects_dict = dict(handle_no_detection(
        directory_path=image_dir, data_dict=noun_objects_dict))
    relation_objects_dict = dict(get_relation_boxes_in_json(
        directory_path=image_dir, data_dict=noun_objects_dict))
    return noun_objects_dict, relation_objects_dict


@pyrallis.wrap()
def evaluate(config: RunConfig, prompts=["a blue cake and a red suitcase"], image_dir="sample_images", input_images_path=["sample_images/a blue cake and a red suitcase.png"], use_api=False, parsed_inputs=None, fine_coeff=None, coarse_coeff=None):

    if use_api:
        api_model_name = config.api_model_name
        api_base_url = config.api_base_url
        api_key = config.api_key
        api_temperature = config.api_temperature
        api_top_p = config.api_top_p
        api_max_tokens = config.api_max_tokens
        api_chat_mode = config.api_chat_mode
        llm_api = LllmApi(api_model_name=api_model_name, api_base_url=api_base_url, api_key=api_key,
                          api_temperature=api_temperature, api_top_p=api_top_p, api_max_tokens=api_max_tokens, api_chat_mode=api_chat_mode)

        decomposition_shots = read_txt_file(config.prompt_decomposition_path)
        question_generation_shots = read_txt_file(
            config.disjoint_question_generation)
        decompose_system_prompt = config.decompose_system_prompt
        question_generation_system_prompt = config.disjoin_system_prompt

        parsed_inputs = []
        for prompt in prompts:
            decompose_pormpt = llm_api.generate_answer(
                system_prompt=decompose_system_prompt, user_prompt=decomposition_shots.format(prompt))
            llm_data = llm_api.generate_answer(
                system_prompt=question_generation_system_prompt, user_prompt=question_generation_shots.format(decompose_pormpt))

            parsed_inputs.append({
                'prompt': prompt,
                'parsed_input': parse_output(llm_data)
            })

    images_file = dict()
    for i, image_path in enumerate(input_images_path):
        image_dir_path, image_file = os.path.split(image_path)
        images_file[i] = image_file
        images_path = image_dir_path

    # reomve yolov9 runs directory to evaluate new images
    if os.path.exists("yolov9/runs"):
        shutil.rmtree("yolov9/runs")

    noun_images, rel_images = object_detection(image_dir=image_dir)
    questions = read_prompt_question_from_api(parsed_inputs)
    meta_data = read_meta_data_question_from_api(parsed_inputs)

    vqa_model = VQAModel()
    nlp = spacy.load("en_core_web_sm")
    sim_image2text = image2text_score(images_path, images_file, noun_images,
                                      questions, vqa_model, nlp, config.colors)
    sim_text2image_noun = text2image_noun_score(
        images_path, images_file, meta_data, noun_images, questions, vqa_model)
    sim_text2image_rel = text2image_rel_score(
        images_path, images_file, meta_data, rel_images, questions, vqa_model)

    sim_vector_noun, sim_vector_noun_questions = flip_matching(
        sim_text2image_noun)
    sim_vector_rel, sim_vector_rel_questions = flip_matching(
        sim_text2image_rel)

    fine_grained_sim_vector = np.vstack(
        [sim_vector_noun, sim_vector_rel]).mean(0).tolist()
    # fine_grained_sim_vector = np.vstack([sim_vector_noun, sim_vector_rel]).max(0).tolist() # max
    coarse_grained_sim_vector = []
    for keys, values in sim_image2text.items():
        coarse_grained_sim_vector.append(np.array(values).mean())

    fine_grained_coeff = fine_coeff if fine_coeff else config.fine_grained_coef
    coarse_grained_coeff = coarse_coeff if coarse_coeff else config.coarse_grained_coef

    sim_vector_questions = {}
    for key, values in sim_image2text.items():
        sim_vector_questions[key] = (coarse_grained_coeff * np.array(sim_image2text[key]) + fine_grained_coeff * (np.vstack(
            [sim_vector_noun_questions[key], sim_vector_rel_questions[key]]).mean(0))) / (fine_grained_coeff + coarse_grained_coeff)
        sim_vector_questions[key] = sim_vector_questions[key].tolist()

    sim_vector = (coarse_grained_coeff * np.array(coarse_grained_sim_vector).T + fine_grained_coeff *
                  np.array(fine_grained_sim_vector)) / (fine_grained_coeff + coarse_grained_coeff)
    sim_vector = sim_vector.tolist()

    return sim_vector, sim_vector_questions


if __name__ == '__main__':
    prompts = ["a blue cake and a red suitcase."]
    image_dir = "sample_images"
    input_images_path = [
        "sample_images/a blue cake and a red suitcase.png"]

    parsed_inputs = [
        {
            "prompt": "a blue cake and a red suitcase.",
            "parsed_input": {
                "assertions": [
                    "there is a blue cake in the image.",
                    "there is a red suitcase in the image.",
                ],
                "questions": [
                    "is there a blue cake in the image?",
                    "is there a red suitcase in the image?",
                ],
                "entities": [
                    "cake",
                    "suitcase",
                ],
                "type": [
                    "noun",
                    "noun",
                ]
            },
            "decompose_pormpt": "Decomposable-Caption: [a blue cake], [a red suitcase].",
        }]

    image_data = dict(use_api=False, parsed_inputs=parsed_inputs,
                      fine_coeff=None, coarse_coeff=None)
    print(evaluate(prompts, image_dir, input_images_path, **image_data))
