# Augment Pattern-Exploiting Training (Aug-PET)

This repository contains the code for [Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676) and our improved version call Aug-PET.


## 🔧 Setup

All requirements for PET can be found in `requirements.txt`. You can install all required packages with `pip install -r requirements.txt`.

## 💬 CLI Usage

The command line interface `cli.py` in this repository currently supports three different training modes (PET, iPET, supervised training), two additional evaluation methods (unsupervised and priming) and 13 different tasks. For Yelp Reviews, AG's News, Yahoo Questions, MNLI and X-Stance, see [the original paper](https://arxiv.org/abs/2001.07676) for further details. For the 8 SuperGLUE tasks, see [this paper](https://arxiv.org/abs/2009.07118).

### Original PET Training and Evaluation

To train and evaluate a PET model for one of the supported tasks, simply run the following command:

    python3 cli.py \
	--method pet \
	--pattern_ids $PATTERN_IDS \
	--data_dir $DATA_DIR \
	--model_type $MODEL_TYPE \
	--model_name_or_path $MODEL_NAME_OR_PATH \
	--task_name $TASK \
	--output_dir $OUTPUT_DIR \
	--do_train \
	--do_eval
    
 where
 - `$PATTERN_IDS` specifies the PVPs to use. For example, if you want to use *all* patterns, specify `PATTERN_IDS 0 1 2 3 4` for AG's News and Yahoo Questions or `PATTERN_IDS 0 1 2 3` for Yelp Reviews and MNLI.
 - `$DATA_DIR` is the directory containing the train and test files (check `tasks.py` to see how these files should be named and formatted for each task).
 - `$MODEL_TYPE` is the name of the model being used, e.g. `albert`, `bert` or `roberta`.
 - `$MODEL_NAME` is the name of a pretrained model (e.g., `roberta-large` or `albert-xxlarge-v2`) or the path to a pretrained model.
 - `$TASK_NAME` is the name of the task to train and evaluate on.
 - `$OUTPUT_DIR` is the name of the directory in which the trained model and evaluation results are saved.
 
You can additionally specify various training parameters for both the ensemble of PET models corresponding to individual PVPs (prefix `--pet_`) and for the final sequence classification model (prefix `--sc_`). For example, the default parameters used for our SuperGLUE evaluation are:
 
 	--pet_per_gpu_eval_batch_size 8 \
	--pet_per_gpu_train_batch_size 2 \
	--pet_gradient_accumulation_steps 8 \
	--pet_max_steps 250 \
	--pet_max_seq_length 256 \
    --pet_repetitions 3 \
	--sc_per_gpu_train_batch_size 2 \
	--sc_per_gpu_unlabeled_batch_size 2 \
	--sc_gradient_accumulation_steps 8 \
	--sc_max_steps 5000 \
	--sc_max_seq_length 256 \
    --sc_repetitions 1
    
For each pattern `$P` and repetition `$I`, running the above command creates a directory `$OUTPUT_DIR/p$P-i$I` that contains the following files:
  - `pytorch_model.bin`: the finetuned model, possibly along with some model-specific files (e.g, `spiece.model`, `special_tokens_map.json`)
  - `wrapper_config.json`: the configuration of the model being used
  - `train_config.json`: the configuration used for training
  - `eval_config.json`: the configuration used for evaluation
  - `logits.txt`: the model's predictions on the unlabeled data
  - `eval_logits.txt`: the model's prediction on the evaluation data
  - `results.json`: a json file containing results such as the model's final accuracy
  - `predictions.jsonl`: a prediction file for the evaluation set in the SuperGlue format
  
The final (distilled) model for each repetition `$I` can be found in `$OUTPUT_DIR/final/p0-i$I`, which contains the same files as described above.

🚨 If your GPU runs out of memory during training, you can try decreasing both the `pet_per_gpu_train_batch_size` and the `sc_per_gpu_unlabeled_batch_size` while increasing both `pet_gradient_accumulation_steps` and `sc_gradient_accumulation_steps`.


### Aug-PET Training and Evaluation

```
python cli.py \
--method pet \
--pattern_ids $PATTERN_IDS \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME_OR_PATH \
--task_name $TASK \
--output_dir $OUTPUT_DIR \
--lm_training \
--do_train \
--do_eval \
--train_examples 10 \
--unlabeled_examples 1000 \
--split_examples_evenly \
--pet_per_gpu_train_batch_size 1 \
--pet_per_gpu_unlabeled_batch_size 3 \
--pet_gradient_accumulation_steps 4 \
--pet_repetitions 1 \
--pet_max_steps 250 \
--sc_per_gpu_train_batch_size 4 \
--sc_per_gpu_unlabeled_batch_size 4 \
--sc_gradient_accumulation_steps 4 \
--sc_max_steps 2500 \
--sc_repetitions 1 \
--overwrite_output_dir \
--augmentation
```



## 📕 Reference

If you make use of the code in this repository, please cite the following papers and code repos:

    @article{schick2020exploiting,
      title={Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference},
      author={Timo Schick and Hinrich Schütze},
      journal={Computing Research Repository},
      volume={arXiv:2001.07676},
      url={http://arxiv.org/abs/2001.07676},
      year={2020}
    }

    @article{schick2020small,
      title={It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners},
      author={Timo Schick and Hinrich Schütze},
      journal={Computing Research Repository},
      volume={arXiv:2009.07118},
      url={http://arxiv.org/abs/2009.07118},
      year={2020}
    }
    
    https://github.com/timoschick/pet
