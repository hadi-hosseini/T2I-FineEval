import os

from tqdm import tqdm
from PIL import Image

from src.vqa import compute_vqa_score
from src.delete_adjectives import find_adjectives_and_delete


# calculate dascore for each pair of full image and questions
def image2text_score(images_path, images_file, noun_images, questions, vqa_model, nlp, colors):
    sim_image2text = {}

    for image_id in tqdm(range(len(noun_images.keys()))):
        sim_image2text[image_id] = []

        image_path = os.path.join(images_path, images_file[image_id])
        img = Image.open(image_path)

        for question_id in range(len(questions[image_id])):
            question = questions[image_id][question_id]
            question = find_adjectives_and_delete(question, colors, nlp)
            score = compute_vqa_score(
                [img], [question], vqa_model)[0]

            sim_image2text[image_id].append(score)

    return sim_image2text

# calculate dascore for each pair of image2text noun


def text2image_noun_score(images_path, images_file, meta_data, noun_images, questions, vqa_model, extend_threshold=0.2):
    sim_text2image_noun = {}

    for image_id in tqdm(range(len(questions.keys()))):
        sim_text2image_noun[image_id] = []

        for question_id in range(len(questions[image_id])):
            question = questions[image_id][question_id]
            sim_text2image_noun[image_id].append([])

            for object_id in range(len(noun_images[image_id]) + 1):
                image_path = os.path.join(images_path, images_file[image_id])
                img = Image.open(image_path)

                if object_id != len(noun_images[image_id]):
                    box_coordinate = noun_images[image_id][object_id][0]

                    width = box_coordinate[2] - box_coordinate[0]
                    height = box_coordinate[3] - box_coordinate[1]

                    extension_w = int(width * extend_threshold)
                    extension_h = int(height * extend_threshold)

                    new_box_coordinate = (
                        max(0, box_coordinate[0] - extension_w),
                        max(0, box_coordinate[1] - extension_h),
                        min(img.width, box_coordinate[2] + extension_w),
                        min(img.height, box_coordinate[3] + extension_h))

                    box = img.crop(new_box_coordinate).resize(img.size)

                else:
                    box = img

                if meta_data[image_id][question_id][0] == "noun":
                    score = compute_vqa_score(
                        [box], [question], vqa_model)[0]
                    score *= 2

                    sim_text2image_noun[image_id][question_id].append(score)
                else:
                    sim_text2image_noun[image_id][question_id].append(0)

    return sim_text2image_noun


# calculate dascore for each pair of image2text relation
def text2image_rel_score(images_path, images_file, meta_data, rel_images, questions, vqa_model, extend_threshold=0.2):
    sim_text2image_rel = {}
    for image_id in tqdm(range(len(questions.keys()))):
        sim_text2image_rel[image_id] = []

        for question_id in range(len(questions[image_id])):
            question = questions[image_id][question_id]
            sim_text2image_rel[image_id].append([])

            for rel_id in range(len(rel_images[image_id]) + 1):
                image_path = os.path.join(images_path, images_file[image_id])
                img = Image.open(image_path)

                if rel_id != len(rel_images[image_id]):
                    box_coordinate = rel_images[image_id][rel_id][0]

                    width = box_coordinate[2] - box_coordinate[0]
                    height = box_coordinate[3] - box_coordinate[1]

                    extension_w = int(width * extend_threshold)
                    extension_h = int(height * extend_threshold)

                    new_box_coordinate = (
                        max(0, box_coordinate[0] - extension_w),
                        max(0, box_coordinate[1] - extension_h),
                        min(img.width, box_coordinate[2] + extension_w),
                        min(img.height, box_coordinate[3] + extension_h))

                    box = img.crop(new_box_coordinate).resize(img.size)

                else:
                    box = img

                if meta_data[image_id][question_id][0] == "relation":
                    score = compute_vqa_score(
                        [box], [question], vqa_model)[0]
                    score *= 2

                    sim_text2image_rel[image_id][question_id].append(score)
                else:
                    sim_text2image_rel[image_id][question_id].append(0)

    return sim_text2image_rel
