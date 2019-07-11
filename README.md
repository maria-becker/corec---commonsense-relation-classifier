### COREC - a neural multi-label COmmonsense RElation Classification system

When using the classifier, please cite the following paper: 

Becker, M., Staniek, M., Nastase, V., Frank, A. (2019):
Assessing the Difficulty of Classifying ConceptNet Relations in a Multi-Label Classification Setting. 
RELATIONS - Workshop on meaning relations between phrases and sentences (co-located with IWCS). 
Gothenburg, Sweden.
https://www.aclweb.org/anthology/W19-0801


### INSTRUCTIONS

Three files are needed to run the neural networks for commonsense relation classification:

### data_reader_mcl.py

data_reader_mlc.py contains the classes DataReader and the class Triple.

The class DataReader takes as an argument the file containing the following: LeftEntity, RightEntity, Relation (Relation2, Relation3... for multi-label instances), separated by \t . It iters over the file
and turns every line into a Triple-object.

The Triple class takes a line formatted as above and simply splits it into its parts, contained in the variables: self.left_term, self.right_term, self.relation.
self.left_term and self.right_term are further tokenized (because many are multi word entities) to create self.tok_left_term and self.tok_right_term.

### model.py

A PyTorch model is written that then needs to be trained.

Only the init-method where all weight matrices and the forward method that composes differentiable functions and the weight matrices together have to be written,
the backpropagation is then automatically done by PyTorch.

### run_neural_net.py

run_neural_net.py contains the whole code necessary to load the dataset (by importing data_reader.py) and reading the word vectors.
For reading the word vectors, it contains a class VectorProcessor which takes as input a vector file and a vocabulary.

The vocabulary is used to only get the needed vectors for the classification data to keep the resulting embedding layer very small.

After loading the word vector file and filtering it, the Vector Processor object then has capabilities to turn words into integers (wordids) with the get_word_ids method or into
vectors with the get_word_vector method.

After loading the dataset and the word embeddings from the file, sklearn.model_selection.train_test_split is then used to create a train, dev and testset from the data.
a certain model is imported from the modelle subdirectory, and the functions train, early_stopping and validate functions are used for the training.

#### train

The train function creates the model object and other important objects like the loss-function and the optimizer.
Then, for multiple iterations, it will iterate over the training data. The VectorProcessor object is then used to get the word ids from the triples,
which then get turned into a PyTorch. LongTensor enveloped by a Pytorch.autograd.Variable. This data is fed into the model and the loss from this
training example is then used to backpropagate and optimize the embeddings.
After every epoch, the validate and early_stopping functions are called.

#### validate

The validate function takes the model object trained until this point and lets the model run over the validation (dev) data. The resulting loss is
then appended to a list saving all the validation losses for each epoch.

#### early_stopping

The early_stopping method takes as input the list of validation losses and an integer n. The last n entries are then analysed.
If the first element is a better result then every next element, the training will then be stopped.
