import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import PIL
from torchvision import transforms
from collections import defaultdict, Counter
from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer, BertModel, AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader
from torchmultimodal.models.flava.model import flava_model_for_classification
from typing import List, Tuple

torch.set_float32_matmul_precision('medium')


class Flava(pl.LightningModule):
    def __init__(self, num_classes: int):
      """
      Initialize the Trainer class.

      Args:
        num_classes (int): The number of classes for classification.
      """
      super().__init__()
      self.flavamodel = flava_model_for_classification(num_classes=num_classes)


    def forward(self, input_ids: torch.Tensor, image: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
      """
      Performs forward pass through the model.

      Args:
        input_ids (Tensor): Input IDs for the text.
        image (Tensor): Input image.
        answers (Tensor): Ground truth answers.

      Returns:
        Tensor: Output of the model.
      """
      return self.flavamodel(text=input_ids, image=image, labels=answers)    
    
    def common_step(self, batch: dict, batch_idx: int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
      """
      Performs a common step in the training/validation loop.

      Args:
        batch (dict): A dictionary containing the input data batch.
        batch_idx (int): The index of the current batch.

      Returns:
        tuple: A tuple containing the logits, loss, and accuracy.
      """
      image = batch['image']
      input_ids = batch['input_ids']
      text = batch['answers']

      # Calls the forward function
      output = self(input_ids, image, text)

      loss = output.loss
      predictions = output.logits.argmax(-1)
      correct = (predictions == text).sum().item()
      accuracy = correct / text.shape[0]

      return output.logits, loss, accuracy

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
      """
      Perform a single training step.

      Args:
        batch: The input batch for training.
        batch_idx: The index of the current batch.

      Returns:
        The loss value for the training step.
      """
      # Define the training step logic here
      _, loss, accuracy = self.common_step(batch, batch_idx)

      self.log('training_loss', loss)
      self.log('training_accuracy', accuracy)

      return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
      """
      Perform a validation step on the given batch.

      Args:
        batch (dict): The input batch for validation.
        batch_idx (int): The index of the current batch.

      Returns:
        torch.Tensor: The loss value for the validation step.
      """
      # Define the validation step logic here
      _, loss, accuracy = self.common_step(batch, batch_idx)

      self.log('validation_loss', loss)
      self.log('validation_accuracy', accuracy)

      return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
      """
      Perform a single step during the testing phase.

      Args:
        batch (dict): The input batch for testing.
        batch_idx (int): The index of the current batch.

      Returns:
        torch.Tensor: The loss value for the current batch.
      """
      # Define the test step logic here
      _, loss, accuracy = self.common_step(batch, batch_idx)

      self.log('test_loss', loss)
      self.log('test_accuracy', accuracy)

      return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
      """
      Configure the optimizer and learning rate scheduler.

      Returns:
        torch.optim.Optimizer: The optimizer to be used for training.
      """
      return torch.optim.AdamW(self.parameters(), lr=5e-5)
    



# load the data
dataset = load_dataset('textvqa', trust_remote_code=True)


with open("vocabs/answers_textvqa_more_than_1.txt") as f:
  vocab = f.readlines()

answer_to_idx = {}
for idx, entry in enumerate(vocab):
  answer_to_idx[entry.strip("\n")] = idx



def image_transform(image: PIL.Image.Image) -> transforms.Compose:
  """
  Apply a series of transformations to the input image.

  Args:
    image (PIL.Image.Image): The input image.

  Returns:
    torchvision.transforms.Compose: A composed transformation object.

  """
  transform_image = transforms.Compose([transforms.ToTensor(), 
                 transforms.Resize([224,224], antialias=True)
                ])
  return transform_image(image)

def transform_text(tokenizer: AutoTokenizer, text: str) -> dict:
  """
  Tokenizes the input text using the given tokenizer.

  Args:
    tokenizer (AutoTokenizer): The tokenizer object to use for tokenization.
    text (str): The input text to be tokenized.

  Returns:
    dict: A dictionary containing the tokenized text, with additional properties like input_ids, attention_mask, etc.
  """
  return tokenizer(text,
           return_tensors='pt',
           padding="max_length",
           max_length=512)

def transform_answers(answer_list):
    """
    Transforms a list of answers into an index of the most frequent answer.
    
    Parameters:
    answer_list (list): A list of answers.

    Returns:
    int: The index of the most frequent answer, or 0 if the list is empty.
    """

    # Handle empty answer list
    if not answer_list:
        return 0

    # Count frequency of each answer
    ans_to_count = Counter(answer_list)

    # Find the answer with the highest frequency
    max_value = max(ans_to_count, key=ans_to_count.get)

    # Get the index of the most frequent answer, defaulting to 0
    ans_idx = answer_to_idx.get(max_value, 0)

    return torch.as_tensor([ans_idx])

def transform(tokenizer: AutoTokenizer, input: dict):
  """
  Transforms the input data into a batch suitable for training.

  Args:
    tokenizer (AutoTokenizer): The tokenizer object used to tokenize the text.
    input (dict): The input data containing the image, question, and answers.

  Returns:
    dict: The transformed batch data.

  """
  batch = {}

  # Transform the image
  batch["image"] = [image_transform(input["image"][0].convert("RGB"))]
  # Transform the question text
  batch.update(transform_text(tokenizer, input['question']))
  # Transform the answers into a class
  batch["answers"] = transform_answers(input["answers"][0])

  return batch



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", 
																					padding='max_length', 
																					max_length=512)
transform = partial(transform, tokenizer)
dataset.set_transform(transform)

# Create the data loader
train_data_loader = DataLoader(dataset['train'], batch_size=128, num_workers=27, shuffle=True)
validation_data_loader = DataLoader(dataset["validation"], batch_size=64, num_workers=27)


# Instantiate the model and trainer
early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)

logger = TensorBoardLogger('tb_logger', name='vqa')

trainer = Trainer(accelerator='gpu', 
                  callbacks=[early_stop_callback],
                  logger=logger,
                  max_epochs=10,
                  check_val_every_n_epoch=1)

# Instantiate the model
model = Flava(len(vocab))


trainer.fit(model, train_data_loader, validation_data_loader)

trainer.validate(model, validation_data_loader)

