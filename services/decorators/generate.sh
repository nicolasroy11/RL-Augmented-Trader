#!/bin/bash
rm -rf generated
mkdir generated

# generates the api code
for filename in ../views/*.py; do
    echo $filename;
    python -m endpoints.cli $filename;
done

# generates the interface definitions
# for filename in ../dto/*.py; do
#     echo $filename;
#     python ./dto/cli.py $filename;
# done

# # generates the enums
# for filename in ../constants/*.py; do
#     echo $filename;
#     python ./enums/cli.py $filename;
# done
