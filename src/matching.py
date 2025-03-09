import numpy as np

# flip matching
def flip_matching(sim_dict):
    sim_scores = []
    sim_scores_questions = {}
    for image_id in list(sim_dict.keys()):
        image_score = np.array(sim_dict[image_id]).max(1)
        sim_scores_questions[image_id] = image_score.tolist()
        image_score = image_score.mean()
        sim_scores.append(image_score)
    return np.array(sim_scores), sim_scores_questions