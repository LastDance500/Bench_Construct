from __future__ import annotations

MAIN_TASKS = [
    {"id": "U1", "split": "1_1", "capability": "Understanding", "name": "Class Definition Understanding", "question_type": "MCQ", "metric": "accuracy"},
    {"id": "U2", "split": "1_2", "capability": "Understanding", "name": "Class Relation Understanding", "question_type": "MCQ", "metric": "accuracy"},
    {"id": "U3", "split": "1_3", "capability": "Understanding", "name": "Property Semantics Understanding", "question_type": "MCQ", "metric": "accuracy"},
    {"id": "U4", "split": "1_4", "capability": "Understanding", "name": "Instance Class Understanding", "question_type": "MCQ", "metric": "accuracy"},
    {"id": "U5", "split": "1_5", "capability": "Understanding", "name": "Instance Description Understanding", "question_type": "MCQ", "metric": "accuracy"},
    {"id": "R1", "split": "2_1", "capability": "Reasoning", "name": "Inferred Class Relation Reasoning", "question_type": "MCQ", "metric": "accuracy"},
    {"id": "R2", "split": "2_2", "capability": "Reasoning", "name": "Property Constraint Reasoning", "question_type": "MCQ", "metric": "accuracy"},
    {"id": "R3", "split": "2_3", "capability": "Reasoning", "name": "Inferred Instance Class Reasoning", "question_type": "MCQ", "metric": "accuracy"},
    {"id": "R4", "split": "2_4", "capability": "Reasoning", "name": "SWRL-based Rule Reasoning", "question_type": "MCQ", "metric": "accuracy"},
    {"id": "R5", "split": "2_5", "capability": "Reasoning", "name": "Description Logic Reasoning", "question_type": "T/FQ", "metric": "accuracy"},
    {"id": "L1", "split": "3_1", "capability": "Learning", "name": "Ontology Term Extraction from Text", "question_type": "Generation", "metric": "entity_f1"},
    {"id": "L2", "split": "3_2", "capability": "Learning", "name": "Class Definition Generation", "question_type": "Generation", "metric": "bertscore"},
    {"id": "L3", "split": "3_3", "capability": "Learning", "name": "Class Hierarchy Construction", "question_type": "Generation", "metric": "triple_f1"},
    {"id": "L4", "split": "3_4", "capability": "Learning", "name": "Property Relation Construction", "question_type": "Generation", "metric": "triple_f1"},
    {"id": "L5", "split": "3_5", "capability": "Learning", "name": "Constraint Construction", "question_type": "Generation", "metric": "triple_f1"},
]

AUXILIARY_TASKS = [
    {"id": "A1", "split": "aux_alignment", "capability": "Auxiliary", "name": "Ontology Alignment", "question_type": "Generation", "metric": "tuple_f1"},
]

TASKS_BY_SPLIT = {task["split"]: task for task in MAIN_TASKS + AUXILIARY_TASKS}
MAIN_SPLITS = tuple(task["split"] for task in MAIN_TASKS)
