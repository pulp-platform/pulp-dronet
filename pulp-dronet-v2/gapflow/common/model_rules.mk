# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.


CNN_GEN = $(MODEL_GEN_SQ8)
CNN_GEN_INCLUDE = $(MODEL_GEN_INCLUDE_SQ8)
CNN_LIB = $(MODEL_LIB_SQ8)
CNN_LIB_INCLUDE = $(MODEL_LIB_INCLUDE_SQ8)


ifdef MODEL_L1_MEMORY
  MODEL_GEN_EXTRA_FLAGS += --L1 $(MODEL_L1_MEMORY)
endif

ifdef MODEL_L2_MEMORY
  MODEL_GEN_EXTRA_FLAGS += --L2 $(MODEL_L2_MEMORY)
endif

ifdef MODEL_L3_MEMORY
  MODEL_GEN_EXTRA_FLAGS += --L3 $(MODEL_L3_MEMORY)
endif


$(MODEL_BUILD):
	mkdir $(MODEL_BUILD)

$(MODEL_ONNX): $(TRAINED_ONNX_MODEL) | $(MODEL_BUILD)
	cp $< $@


$(MODEL_STATE): $(MODEL_ONNX) $(IMAGES) $(NNTOOL_SCRIPT_DEPLOY) | $(MODEL_BUILD)
	echo "GENERATING NNTOOL STATE FILE"
	echo $(MODEL_BUILD)
	$(NNTOOL) -s $(NNTOOL_SCRIPT_DEPLOY) $< -q

nntool_model_evaluation: $(MODEL_ONNX) $(IMAGES) $(NNTOOL_SCRIPT_EVAL) | $(MODEL_BUILD)
	echo "Computing testing scores"
	echo $(MODEL_BUILD)
	$(NNTOOL) -s $(NNTOOL_SCRIPT_EVAL) $< -q

nntool_state: $(MODEL_STATE)

# Runs NNTOOL with its state file to generate the autotiler model code
$(NNTOOL_MODEL_DIR)/$(MODEL_SRC): $(MODEL_STATE) $(MODEL_ONNX) | $(MODEL_BUILD)
	echo "GENERATING AUTOTILER MODEL"
	$(NNTOOL) -g -M $(NNTOOL_MODEL_DIR) -m $(MODEL_SRC) -T $(TENSORS_DIR) $<

nntool_gen: $(NNTOOL_MODEL_DIR)/$(MODEL_SRC)

# Build the code generator from the model code
$(MODEL_GEN_EXE): $(CNN_GEN) $(NNTOOL_MODEL_DIR)/$(MODEL_SRC) | $(MODEL_BUILD)
	echo "COMPILING AUTOTILER MODEL"
	gcc -g -o $(MODEL_GEN_EXE) -I. -I$(TILER_INC) -I$(TILER_EMU_INC) $(CNN_GEN_INCLUDE) $(CNN_LIB_INCLUDE) $^ $(TILER_LIB) $(SDL_FLAGS) 

compile_model: $(MODEL_GEN_EXE)

# Run the code generator to generate GAP graph and kernel code
$(MODEL_GEN_C): $(MODEL_GEN_EXE)
	echo "RUNNING AUTOTILER MODEL"
	$(MODEL_GEN_EXE) -o $(MODEL_BUILD) -c $(MODEL_BUILD) $(MODEL_GEN_EXTRA_FLAGS)

# A phony target to simplify including this in the main Makefile
model: $(MODEL_GEN_C)

clean_model:
	$(RM) $(MODEL_GEN_EXE)
	$(RM) -rf $(MODEL_BUILD)
	$(RM) -rf $(NNTOOL_MODEL_DIR) 

.PHONY: model clean_model clean_train train nntool_gen nntool_state compile_model clean_nntool
