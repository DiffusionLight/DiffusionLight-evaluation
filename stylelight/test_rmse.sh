test_data_path=$1
ground_truth_path=$2
python fast_rmse.py --fake $test_data_path/mirror/ --real $ground_truth_path/mirror/
python fast_rmse.py --fake $test_data_path/matte_silver/ --real $ground_truth_path/matte_silver/
python fast_rmse.py --fake $test_data_path/diffuse/ --real $ground_truth_path/diffuse/
