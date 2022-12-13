import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image

if __name__ == '__main__':

    # Read csv containing the clinical parameters and image paths

    csv_path = 'examples/example.csv'
    df_input = pd.read_csv(csv_path)

    # Load models

    images_model = tf.keras.models.load_model('models/images_model')

    with open('models/clinical_model', "rb") as f:
        clinical_model = pickle.load(f)

    with open('models/hybrid_model', "rb") as f:
        hybrid_model = pickle.load(f)

    # Create feature extractor model from the CNN used to classify the images

    feature_extractor_model = tf.keras.Model(inputs=images_model.inputs, outputs=images_model.layers[-4].output)
    feature_extractor_model.compile(loss='categorical_crossentropy',
                                    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9))

    # Load images to classify

    im_list = []

    for im_path in df_input['ImagePath'].values:

        im = tf.keras.preprocessing.image.load_img(im_path)

        # Images are resized to 200 by 200 pxls

        im = tf.keras.preprocessing.image.img_to_array(im.resize([200, 200], resample=Image.Resampling.BILINEAR))

        # Normalize images to have zero mean and unit variance

        im -= im.mean()
        im /= im.std()

        im_list.append(im)

    im_list = np.array(im_list)

    # Extract all features used for classification

    image_features = feature_extractor_model.predict(im_list, batch_size=1)
    clinical_features = df_input[df_input.columns[-5:]].values
    hybrid_features = np.concatenate((clinical_features, image_features), axis=1)

    # Load scalers for data normalization

    with open('scalers', "rb") as f:
        clinical_scaler = pickle.load(f)
        hybrid_scaler = pickle.load(f)

    # Classify

    image_prediction = images_model.predict(im_list, batch_size=1)
    image_prediction = np.argmax(image_prediction, axis=1).astype(int)

    clinical_prediction = clinical_model.predict(clinical_scaler.transform(clinical_features))
    clinical_prediction = np.argmax(clinical_prediction, axis=1).astype(int)

    hybrid_prediction = hybrid_model.predict(hybrid_scaler.transform(hybrid_features))
    hybrid_prediction = np.argmax(hybrid_prediction, axis=1).astype(int)

    # Print classification results

    table_captions = ['Ischemic Colitis', 'IBD', 'Infectious Colitis']

    print('\n\n%20s | %20s | %20s | %20s ' % ('ID', 'Image Prediction',
                                              'Clinical Prediction',
                                              'Hybrid Prediction',))
    print('='*89)

    for case_id, ipred, cpred, hpred in zip(df_input['ID'], image_prediction, clinical_prediction, hybrid_prediction):

        if not isinstance(case_id, str):
            case_id = str(case_id)

        print('%20s | %20s | %20s | %20s ' % (case_id, table_captions[ipred],
                                              table_captions[cpred], table_captions[hpred]))
