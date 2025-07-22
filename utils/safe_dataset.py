import traceback
from torch.utils.data import Dataset

class SafeDataset(Dataset):
    """
    A wrapper around a Dataset that patches the __getitem__ method to catch exceptions
    and return None if an exception occurs, assuming that it happened because the file
    is corrupted or missing and not because of a bug in the code.

    Attributes:
        dataset (Dataset): The original PyTorch Dataset to be wrapped.
        disable (bool): If True, disables the wrapper's try-except.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves an item from the wrapped dataset at the given index.
        If an exception occurs, tries to get another sample until successful.

        Args:
            index (int): The index of the item to be retrieved.

        Returns:
            Any: The item from the wrapped dataset, trying alternative indices if corrupted.
        """

        max_retries = 10  # Prevent infinite loops
        attempted_indices = set()
        current_index = index
        
        for retry in range(max_retries):
            # Avoid trying the same index twice
            if current_index in attempted_indices:
                # Find a new index we haven't tried yet
                import random
                current_index = random.randint(0, len(self.dataset) - 1)
                continue
                
            attempted_indices.add(current_index)
            
            try:
                item = self.dataset[current_index]
                
                # Log if we had to use an alternative sample
                if current_index != index:
                    print(f"Successfully loaded alternative sample at index {current_index} (original index {index} was corrupted)")
                
                return item
            except Exception as e:
                print(f"Error at index {current_index}, trying another sample. \nException: {e}\n{traceback.format_exc()}")
                # Try next index (with wraparound)
                current_index = (current_index + 1) % len(self.dataset)
        
        # If we've exhausted retries, raise an exception
        raise RuntimeError(f"Failed to load any sample after {max_retries} attempts starting from index {index}")
