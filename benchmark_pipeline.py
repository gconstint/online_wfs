
import timeit
from pipline import task
from core import calculate_wavelength



def run_pipeline_task():
    """Wrapper function to run the pipeline task for timeit."""


    params = dict()

    # Detector Configuration
    params['pixel_size'] = [0.715e-6, 0.715e-6]  # Pixel size in meters (x, y)
    params['wavelength'] = calculate_wavelength(7100)  # X-ray wavelength (eV to m)

    # Optical System Geometry
    params['det2sample'] = 0.35  # Grating-to-detector distance (m)
    params['total_dist'] = 6.5  # Source-to-detector distance (m)
    params['source_dist'] = params['total_dist'] - params['det2sample']  # Source-to-grating distance (m)

    # Grating Parameters
    period = 18.38e-6  # Base grating period (m)
    params['grating_period'] = period / 2  # Effective grating period
    # Calculate expected self-imaging period based on geometry
    params['pattern_period'] = params['grating_period'] * params['total_dist'] / params['source_dist']

    # Data Source Configuration
    img_path = "sample_exp.tif"
    params.update({
        'image_path': img_path,
        "dark_image_path": None,  # Optional dark field correction
        "flat_image_path": None,  # Optional flat field correction
    })

    # The task is a generator, so we need to iterate through it to execute it
    for _ in task(params):
        pass

if __name__ == "__main__":


    # Number of times to run the test
    number_of_runs = 10
    
    # Time the execution
    execution_time = timeit.timeit(run_pipeline_task, number=number_of_runs)
    
    print(f"Average runtime over {number_of_runs} runs: {execution_time / number_of_runs:.4f} seconds")
