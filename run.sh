START=0
END=32



python create_principle_sents_per_file_gcs.py \
--input_pattern "sent_eq_4k_25/en_nd1k_ml10k/*.jsonl" \
--output_dir "sent_score/en_nd1k_ml10k" \
--num_processes 224 \
--start $START \
--end $END \
--tokenizer t5-base

python create_principle_sents_per_file_gcs.py \
--input_pattern "sent_eq_4k_25/ko_corpora/*.jsonl" \
--output_dir "sent_score/ko_corpora" \
--num_processes 224 \
--start $START \
--end $END \
--tokenizer "KETI-AIR/ke-t5-base"

python create_principle_sents_per_file_gcs.py \
--input_pattern "sent_eq_4k_25/en_subset/*.jsonl" \
--output_dir "sent_score/en_subset" \
--num_processes 224 \
--start $START \
--end $END \
--tokenizer "KETI-AIR/ke-t5-base"
