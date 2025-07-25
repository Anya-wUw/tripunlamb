#!/bin/bash
set -e

bash /mnt/extremessd10tb/borisiuk/open-unlearning/scripts/popqa_finetune.sh
bash /mnt/extremessd10tb/borisiuk/open-unlearning/scripts/popqa_unlearn1.sh
bash /mnt/extremessd10tb/borisiuk/open-unlearning/scripts/popqa_unlearn_eval1.sh