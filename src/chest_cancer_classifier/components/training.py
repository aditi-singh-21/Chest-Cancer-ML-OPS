import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from chest_cancer_classifier.config.configuration import TrainModelConfig

class Training:
    def __init__(self, config: TrainModelConfig):
        self.config = config
        
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
    def train_valid_gen(self):
        
        datagen_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )
        
        dataflow_kwargs = dict(
            target_size=self.config.image_size_params[:-1],
            batch_size=self.config.batch_size_params,
            interpolation="bilinear"
        )
        
        valid_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagen_kwargs
        )
        
        self.valid_gen = valid_data_gen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        
        if self.config.is_augmentation_params:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagen_kwargs
            )
        else:
            train_datagen = valid_data_gen
            
        self.train_gen = train_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        
    def train(self):
        self.steps_per_epoch = self.train_gen.samples // self.train_gen.batch_size
        self.validation_steps = self.valid_gen.samples // self.valid_gen.batch_size

        # Callbacks for learning rate reduction and early stopping
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )

        self.model.fit(
            self.train_gen,
            epochs=self.config.epochs_params,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_gen,
            callbacks=[lr_reduce, early_stop]
        )
        
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

