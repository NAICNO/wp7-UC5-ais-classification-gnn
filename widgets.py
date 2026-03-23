"""
Jupyter ipywidgets for AIS Graph Classification demonstrator.

Provides interactive widgets for model selection, hyperparameter tuning,
and execution mode control in Jupyter notebooks.

Follows the NAIC widget pattern with helper functions,
tuple-based build_widgets, and individual-widget get_args_from_widgets.
"""

import ipywidgets as widgets
import argparse


def create_dropdown(options, value, description):
    """Helper function to create a Dropdown widget."""
    return widgets.Dropdown(options=options, value=value, description=description)


def create_int_slider(value, min_val, max_val, step, description):
    """Helper function to create an IntSlider widget."""
    return widgets.IntSlider(value=value, min=min_val, max=max_val, step=step, description=description)


def create_float_slider(value, min_val, max_val, step, description, readout_format='.4f'):
    """Helper function to create a FloatSlider widget."""
    return widgets.FloatSlider(
        value=value, min=min_val, max=max_val, step=step,
        description=description, readout_format=readout_format,
    )


def create_int_input(value, description):
    """Helper function to create an IntText widget."""
    return widgets.IntText(value=value, description=description)


def build_widgets():
    """
    Build all widgets for AIS graph classification experiment configuration.

    Returns:
        Tuple of widgets: (model_type, learning_rate, epochs, batch_size,
                           patience, hidden_dim, depth, seed)
    """
    model_type_widget = create_dropdown(
        ['GCN', 'GSG', 'GAT'],
        'GCN',
        'Model:',
    )
    learning_rate_widget = create_float_slider(0.01, 0.001, 0.1, 0.001, 'Learning Rate:')
    epochs_widget = create_int_slider(100, 10, 1000, 10, 'Epochs:')
    batch_size_widget = create_int_slider(600, 100, 4000, 100, 'Batch Size:')
    patience_widget = create_int_slider(20, 5, 100, 5, 'Patience:')
    hidden_dim_widget = create_int_slider(64, 16, 256, 16, 'Hidden Dim:')
    depth_widget = create_int_slider(3, 1, 6, 1, 'Depth:')
    seed_widget = create_int_input(0, 'Seed:')

    return (
        model_type_widget,
        learning_rate_widget,
        epochs_widget,
        batch_size_widget,
        patience_widget,
        hidden_dim_widget,
        depth_widget,
        seed_widget,
    )


def create_execution_mode_dropdown():
    """
    Creates a dropdown widget for selecting the execution mode.

    Returns:
        A Dropdown widget for selecting the execution mode.
    """
    execution_mode_dropdown = create_dropdown(
        ['Train All Models', 'Train Single Model', 'Evaluate Only', 'No Run'],
        'No Run',
        'Execution Mode:',
    )

    def handle_dropdown_change(change):
        config_map = {
            'Train All Models': 'Training GCN, GSG, and GAT with all learning rates',
            'Train Single Model': 'Training selected model with selected parameters',
            'Evaluate Only': 'Evaluating saved models from results/',
            'No Run': 'Skip training, go to analysis of existing results',
        }
        custom_variable = config_map.get(change.new, 'No valid option selected')
        print(custom_variable)

    execution_mode_dropdown.observe(handle_dropdown_change, names='value')
    return execution_mode_dropdown


def get_args_from_widgets(
    model_type_widget,
    learning_rate_widget,
    epochs_widget,
    batch_size_widget,
    patience_widget,
    hidden_dim_widget,
    depth_widget,
    seed_widget,
):
    """
    Convert individual widget values to an argparse.Namespace.

    Returns:
        argparse.Namespace with all experiment parameters.
    """
    return argparse.Namespace(
        model=model_type_widget.value,
        lr=learning_rate_widget.value,
        epochs=epochs_widget.value,
        batch_size=batch_size_widget.value,
        patience=patience_widget.value,
        hidden=hidden_dim_widget.value,
        depth=depth_widget.value,
        seed=seed_widget.value,
    )


def display_widgets(widgets_tuple, exec_widget=None):
    """
    Display all widgets in a VBox layout.

    Args:
        widgets_tuple: Tuple of widgets returned by build_widgets.
        exec_widget: Optional execution mode widget to append.

    Returns:
        A VBox containing all widgets.
    """
    items = list(widgets_tuple)
    if exec_widget is not None:
        items.append(exec_widget)
    return widgets.VBox(items)
