from sklearn.model_selection import GroupShuffleSplit
import logging


"""
Splits labels into train, test, and val
"""
class LabelSplitter():
    def __init__(self):
        pass

    """
    Splits labels dataframe into train, test, and validation dataframes
    ================== ===========================================================================
    **Arguments:**
    labels_df          labels dataframe

    **Returns:**
    train_df           train dataframe
    test_df            test dataframe
    val_df             validation dataframe
    ================== ===========================================================================
    """
    def split_labels(self, labels_df):
        # Use group shuffle split to split the dataframe into train, test and validation sets.
        # Group shuffle split returns indices associated with the splits.
        # The groups option ensures that the entire subject is assigned to one of test, train, or val.
        # This is done so the machine doesn't memorize a subject and perform well on test or val simply
        # because it's seen a similar image from the same subject during training.
        # This would result in poor generalizability
        train_inds, test_inds = next(GroupShuffleSplit(test_size=.15, n_splits=2, random_state=9).
                                     split(labels_df, groups=labels_df['subject']))

        test_df = labels_df.iloc[test_inds].reset_index(drop=True)
        test_df.to_csv('labels/test.csv', index=False)

        train_df_1 = labels_df.iloc[train_inds].reset_index(drop=True)

        train_inds, val_inds = next(GroupShuffleSplit(test_size=.15, n_splits=2, random_state=4).
                                    split(train_df_1, groups=train_df_1['subject']))

        train_df = train_df_1.iloc[train_inds].reset_index(drop=True)
        train_df.to_csv('labels/train.csv', index=False)

        val_df = train_df_1.iloc[val_inds].reset_index(drop=True)
        val_df.to_csv('labels/val.csv', index=False)

        self.print_distributions(train_df, test_df, val_df)

        return train_df, test_df, val_df

    """
    Prints image and subject distributions for train, test, and validation dataframes
    ================== ===========================================================================
    **Arguments:**
    train_df           train dataframe
    test_df            test dataframe
    val_df             validation dataframe
    ================== ===========================================================================
    """
    def print_distributions(self, train_df, test_df, val_df):
        logging.debug("Number of training images: " + str(train_df.shape[0]))
        logging.debug("Number of validation images: " + str(val_df.shape[0]))
        logging.debug("Number of test images: " + str(test_df.shape[0]))
        logging.debug("")

        logging.debug("Number of training subjects: " + str(train_df['subject'].nunique()))
        logging.debug("Number of validation subjects: " + str(val_df['subject'].nunique()))
        logging.debug("Number of test subjects: " + str(test_df['subject'].nunique()))
        logging.debug("")

        for phase in ['train', 'test', 'val']:
            if phase is 'train':
                data_df = train_df

            elif phase is 'test':
                data_df = test_df

            elif phase is 'val':
                data_df = val_df

            else:
                return

            totalImages = data_df.shape[0]
            gradeZeroImages = data_df[(data_df['0'] == 1)].shape[0]
            gradeOneImages = data_df[(data_df['1'] == 1)].shape[0]
            gradeTwoImages = data_df[(data_df['2'] == 1)].shape[0]
            gradeThreeImages = data_df[(data_df['3'] == 1)].shape[0]

            logging.debug(f"% of {phase} images by tumor classification")
            logging.debug(f"Grade Zero: {gradeZeroImages / totalImages * 100:.2f} ")
            logging.debug(f"Grade One: {gradeOneImages / totalImages * 100:.2f} ")
            logging.debug(f"Grade Two: {gradeTwoImages / totalImages * 100:.2f} ")
            logging.debug(f"Grade Three: {gradeThreeImages / totalImages * 100:.2f} ")
            logging.debug("")

            totalImages = data_df['subject'].nunique()

            gradeZeroImages = data_df[(data_df['0'] == 1)]['subject'].nunique()
            gradeOneImages = data_df[(data_df['1'] == 1)]['subject'].nunique()
            gradeTwoImages = data_df[(data_df['2'] == 1)]['subject'].nunique()
            gradeThreeImages = data_df[(data_df['3'] == 1)]['subject'].nunique()

            logging.debug(f"% of {phase} subjects by tumor classification")
            logging.debug(f"Grade Zero: {gradeZeroImages / totalImages * 100:.2f} ")
            logging.debug(f"Grade One: {gradeOneImages / totalImages * 100:.2f} ")
            logging.debug(f"Grade Two: {gradeTwoImages / totalImages * 100:.2f} ")
            logging.debug(f"Grade Three: {gradeThreeImages / totalImages * 100:.2f} ")
            logging.debug("")
