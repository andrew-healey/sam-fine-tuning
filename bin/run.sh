#!/bin/bash
err_file=".%%%%err.txt"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TRAIN_DIR=$SCRIPT_DIR/..
write_error() {
    echo Encountered error.
    ERR_MSG="$(cat $err_file)"
    cat $err_file
    # python3 src/utils/write_error_to_firebase.py "$ERR_MSG"
    rm $err_file
    exit 1
}
trap 'write_error' ERR
eval 'python $SCRIPT_DIR/run_and_catch_error.py $1 $err_file';
echo "Finished training without error"