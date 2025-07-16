import h5py
import json
import os
import numpy as np


def modify_h5_file_directly(input_path, output_path):
    """
    Directly modify the H5 file by removing the 'groups' parameter from the model config.
    """
    # Copy the original file to avoid modifying it
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found")

    # Copy the file to the output path
    import shutil
    shutil.copy2(input_path, output_path)

    # Open the copied file in read/write mode
    with h5py.File(output_path, 'r+') as f:
        # Get the model config
        if 'model_config' in f.attrs:
            model_config = f.attrs.get('model_config')

            # Convert bytes to string if needed
            if isinstance(model_config, bytes):
                model_config_str = model_config.decode('utf-8')
            else:
                model_config_str = model_config

            # Parse the JSON
            config_dict = json.loads(model_config_str)

            # Function to recursively remove 'groups' from any layer config
            def remove_groups_parameter(config):
                if isinstance(config, dict):
                    # Remove 'groups' from the current level if it exists
                    if 'config' in config and isinstance(config['config'], dict) and 'groups' in config['config']:
                        print(
                            f"Removing 'groups' parameter from layer: {config.get('config', {}).get('name', 'unnamed')}")
                        del config['config']['groups']

                    # Recursively process all dictionary values
                    for key, value in config.items():
                        if isinstance(value, (dict, list)):
                            config[key] = remove_groups_parameter(value)

                elif isinstance(config, list):
                    # Process all items in the list
                    for i, item in enumerate(config):
                        config[i] = remove_groups_parameter(item)

                return config

            # Apply the function to remove 'groups' parameter
            modified_config = remove_groups_parameter(config_dict)

            # Convert back to JSON string
            modified_config_str = json.dumps(modified_config)

            # Update the attribute in the file
            del f.attrs['model_config']  # Delete the old attribute
            f.attrs['model_config'] = modified_config_str  # Add the new one

            print(f"Modified model config saved to {output_path}")
            return True
        else:
            print("No model_config attribute found in the H5 file")
            return False


def extract_model_topology(model_path):
    """
    Extract the model topology to help diagnose issues.
    """
    print(f"Analyzing model file: {model_path}")
    with h5py.File(model_path, 'r') as f:
        # Print the top-level keys
        print("Top-level keys:", list(f.keys()))

        # Print attributes
        print("Attributes:", list(f.attrs.keys()))

        # If there's a model_config attribute, print its structure
        if 'model_config' in f.attrs:
            model_config = f.attrs.get('model_config')
            if isinstance(model_config, bytes):
                model_config_str = model_config.decode('utf-8')
            else:
                model_config_str = model_config

            # Parse the JSON and print a summary
            config_dict = json.loads(model_config_str)
            if 'class_name' in config_dict:
                print(f"Model class: {config_dict['class_name']}")

            # Find all unique layer types
            layer_types = set()

            def collect_layer_types(config):
                if isinstance(config, dict) and 'class_name' in config:
                    layer_types.add(config['class_name'])

                if isinstance(config, dict):
                    for key, value in config.items():
                        if isinstance(value, (dict, list)):
                            collect_layer_types(value)
                elif isinstance(config, list):
                    for item in config:
                        collect_layer_types(item)

            collect_layer_types(config_dict)
            print("Layer types in model:", sorted(layer_types))

            # Count DepthwiseConv2D layers
            depthwise_count = 0

            def count_depthwise_layers(config):
                nonlocal depthwise_count
                if isinstance(config, dict) and config.get('class_name') == 'DepthwiseConv2D':
                    depthwise_count += 1
                    # Print the config of the first DepthwiseConv2D layer
                    if depthwise_count == 1:
                        print("First DepthwiseConv2D layer config:", json.dumps(config.get('config', {}), indent=2))

                if isinstance(config, dict):
                    for key, value in config.items():
                        if isinstance(value, (dict, list)):
                            count_depthwise_layers(value)
                elif isinstance(config, list):
                    for item in config:
                        count_depthwise_layers(item)

            count_depthwise_layers(config_dict)
            print(f"Number of DepthwiseConv2D layers: {depthwise_count}")

        # Check if the model has weights
        if 'model_weights' in f:
            print("Model contains weights")
            # Print the first few weight names
            weight_names = list(f['model_weights'].keys())
            print(f"First 5 weight names (of {len(weight_names)} total):", weight_names[:5])
        else:
            print("No model_weights found")


if __name__ == "__main__":
    model_path = "keras_model.h5"
    fixed_model_path = "keras_model_fixed.h5"

    # First, extract information about the model
    try:
        extract_model_topology(model_path)
    except Exception as e:
        print(f"Error analyzing model: {e}")

    # Then try to modify the file directly
    try:
        success = modify_h5_file_directly(model_path, fixed_model_path)
        if success:
            print("Model file modified successfully. Try loading the modified model.")
            print("If loading still fails, consider recreating the model architecture from scratch.")
        else:
            print("Failed to modify the model file.")
    except Exception as e:
        print(f"Error modifying model file: {e}")