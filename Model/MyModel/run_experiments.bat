@REM ########################
@REM ### MAIN EXPERIMENTS ###
@REM ########################

@REM #################################################################
@REM # NUM TOPICS
@REM python main.py --num_epoch 1000 --trans_induc 'transductive' --num_topics 64 --degree 2
@REM python main.py --num_epoch 1000 --trans_induc transductive --num_topics 32 --degree 2
@REM python main.py --num_epoch 1000 --trans_induc transductive --num_topics 10 --degree 2

@REM @REM #################################################################
@REM python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 1
@REM python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2
@REM python main.py --num_epoch 50 --num_topics 20 --degree 3 --num_hop 1
@REM python main.py --num_epoch 50 --num_topics 20 --degree 1 --num_hop 1

@REM #################################################################
@REM # cosine tfidf
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0    --doc_sim_metric cosine_tfidf
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.51 --doc_sim_metric cosine_tfidf
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.52 --doc_sim_metric cosine_tfidf
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.53 --doc_sim_metric cosine_tfidf
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.54 --doc_sim_metric cosine_tfidf
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.55 --doc_sim_metric cosine_tfidf

@REM #################################################################
@REM # cosine count
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0    --doc_sim_metric cosine_count
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.51 --doc_sim_metric cosine_count
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.52 --doc_sim_metric cosine_count
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.53 --doc_sim_metric cosine_count
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.54 --doc_sim_metric cosine_count
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.55 --doc_sim_metric cosine_count

@REM #################################################################
@REM # jaccard
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0    --doc_sim_metric jaccard
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.51 --doc_sim_metric jaccard
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.52 --doc_sim_metric jaccard
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.53 --doc_sim_metric jaccard
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.54 --doc_sim_metric jaccard
python main.py --num_epoch 50 --num_topics 20 --degree 2 --num_hop 2 --adj_cut_off 0.55 --doc_sim_metric jaccard