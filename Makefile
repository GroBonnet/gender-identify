PART=ENSTA-l40s #ENSTA-h100 #ENSTA-l40s
TIME=01:00:00

# SOURCE = data/img_align_celeba/000615.jpg
# TARGET= data/img_align_celeba/000715.jpg

# PARAMS = --source=$(SOURCE)\
# 	--target=$(TARGET)
	 
run:
	PYTHONPATH=$(PWD) srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py