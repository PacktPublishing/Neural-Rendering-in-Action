import os
import re
import sys
import subprocess
import json
import OpenImageIO as oiio 

def html_content(image_list):
    html_content = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Comparison</title>
        <style>
            .container {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }
            .row {
                display: flex;
                justify-content: space-between;
                gap: 20px; /* Space between images */
                margin-bottom: 20px; /* Space between rows */
            }
            .column {
                text-align: center;
                font-family: Arial, sans-serif;
            }
            img {
                max-width: 300px;  /* Limit image size */
                height: auto;
                border-radius: 8px;
                display: block;
                margin: 0 auto;
            }
            h2 {
                text-align: center;
            }
            p {
                font-size: 14px;
                color: #555;
            }
            .title {
                font-weight: bold;
                font-size: 18px;
                text-align: center;
                margin-top: 10px;
            }
            .message {
                font-size: 16px;
                text-align: center;
                margin-top: 10px;
            }
            .title {
                font-weight: bold;
                font-size: 18px;
                text-align: center;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>

        <div class="container">
    """

    for i, (title, message, path1, path2) in enumerate(image_list):
        filename1 = os.path.basename(path1)
        filename2 = os.path.basename(path2)

        html_content += f"""
            <p class="title">{title}</p>
            <div class="row">
                <div class="column">
                    <img src="{path1}" alt="{filename1}">
                    <p>{filename1}</p>
                </div>
                <div class="column">
                    <img src="{path2}" alt="{filename2}">
                    <p>"Golden image"</p>
                </div>
            </div>
            <p class="message">{message}</p>
        """

    # Close HTML structure
    html_content += """
            </div>
        </div>

    </body>
    </html>
    """
    return html_content



# Define a list to store the extracted tuples
def failing_and_warning_list(reportname, cwd=None):
    failing_and_warning_list = []
    if cwd is None:
        # Use the current directory if not provided
        cwd = os.path.dirname(__file__)
    with open(reportname, "r") as file:
        for line in file:
            # Use regex to find lines that match the pattern
            match = re.search(r'Failed \(Comparing ([^ ]+) to ([^ ]+), Failing pixels:(\d+)%', line)
            if match:
                path1 = match.group(1)
                _, title = os.path.split(path1)
                if not os.path.isabs(path1):
                    path1 = os.path.join(cwd, path1)
                path2 = match.group(2)
                if not os.path.isabs(path2):
                    path2 = os.path.join(cwd, path2)
                msg = "Comparing to HGI baseline image. Failing pixels: " + match.group(3) + "%."
                failing_and_warning_list.append((title, msg, path1, path2))
                continue
            match = re.search(r'No baseline image \(Comparing ([^ ]+)', line)
            if match:
                path1 = match.group(1)
                _, title = os.path.split(path1)
                path2 = path1.replace("HGI", "DirectX")
                path2 = path2.replace("./OutputImages/", "")
                path2 = path2.replace("./", "")
                path2 = os.path.dirname(__file__) + "/../Tests/Aurora/BaselineImages/" + path2
                if not os.path.isfile(path2):
                    path2 = path1
                    msg = "No baseline image. \n"
                    failing_and_warning_list.append((title, msg, path1, path2))
                    continue
                if not os.path.isabs(path1):
                    path1 = os.path.join(cwd, path1)
                msg = "No baseline image, using DX instead. \n"
                buf_captured = oiio.ImageBuf(path1)
                buf_golden   = oiio.ImageBuf(path2)
                total_pixels = oiio.ImageInput.open(path1).spec().width * oiio.ImageInput.open(path1).spec().height
                if buf_captured and buf_golden:
                    # failthresh, warnthresh, failrelative=0.0, warnrelative=0.0, roi=ROI.All, nthreads=0
                    comp = oiio.ImageBufAlgo.compare (buf_captured, buf_golden, 0.1, 0.025)
                    if comp.nfail <= 0.05 :
                        msg += "Images match within tolerance. "
                    else :
                        msg += "Failing pixels: " + "%.2f" % (comp.nfail / total_pixels * 100) + "%. Warning pixels: " + "%.2f" % (comp.nwarn / total_pixels * 100) + "%. "
                    failing_and_warning_list.append((title, msg, path1, path2))
    return failing_and_warning_list

def compare_images(aurora_scenes_items):
    print("Comparing images\n")
    image_list = []
    for aurora_scene_name, aurora_scene_prop in aurora_scenes_items:
        print("Comparing ", aurora_scene_name)
        buf_captured = oiio.ImageBuf(aurora_scene_prop["output"])
        buf_golden   = oiio.ImageBuf(aurora_scene_prop["reference"])
        # failthresh, warnthresh, failrelative=0.0, warnrelative=0.0, roi=ROI.All, nthreads=0
        comp = oiio.ImageBufAlgo.compare (buf_captured, buf_golden, 0.1, 0.025)
        if comp.nwarn <= 0.2 and comp.nfail <= 0.05 :
            message = "Images match within tolerance. "
        else :
            total_pixels = 1#buf_captured.width * buf_captured.height
            message = str(comp.nfail / total_pixels) + " failures, " + str(comp.nwarn / total_pixels) + " warnings. "
            message += "Average error was " + str(comp.meanerror) + ". "
            message += "RMS error was " + str(comp.rms_error) + ". "
            message += "PSNR was " + str(comp.PSNR) + ". "
        benchmark_output_file = aurora_scene_prop["stdout"]
        with open(benchmark_output_file, "r") as out_file:
            for line in out_file:
                # Use regex to find lines that match the pattern
                match = re.search(r'Rendering completed in (\d+)', line)
                if match:
                    t = match.group(1)
                    message += "\nRendering completed in " + str(t) + " ms."
        image_list.append(("CTP " + aurora_scene_name, message, aurora_scene_prop["output"], aurora_scene_prop["reference"]))

    return image_list

def run_plasma_ctp(aurora_scenes_items):
    for _, aurora_scene_prop in aurora_scenes_items:
        my_env = os.environ.copy()
        benchmark_output_file = aurora_scene_prop["stdout"]
        with open(benchmark_output_file, "a") as out_file:
            subprocess.run([plasma_app, "--scene", aurora_scene_prop["scene"], "--output", aurora_scene_prop["output"], "--camera", aurora_scene_prop["camera"], "--output_spp", str(1000)], env=my_env, stdout=out_file)    
    
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python generate_html_report.py <path_to_Plasma> <path_to_aurora_scenes.json> <path_to_report.txt>")
        exit()
    plasma_app = sys.argv[1]
    if not os.path.isfile(plasma_app):
        print("Plasma app not found at: " + plasma_app)
        exit()

    aurora_scenes_json_path = sys.argv[2]
    with open(aurora_scenes_json_path, "r") as json_file:
        aurora_scenes = json.load(json_file)
    aurora_scenes_items = aurora_scenes.items()
    # run_plasma_ctp(aurora_scenes_items)
    image_list = compare_images(aurora_scenes_items)

    my_env = os.environ.copy()
    # result = subprocess.run([test_app], env=my_env, capture_output=True)
    # print (result.stdout)
    # print (result.stderr)

    # Write to an HTML file
    report_path = sys.argv[3]
    plasma_dir = os.path.dirname(plasma_app)
    if not os.path.isabs(report_path):
        report_path = os.path.join(plasma_dir, report_path)
    failing_and_warning_list = failing_and_warning_list(report_path, cwd=plasma_dir)
    image_list += failing_and_warning_list
    html_content = html_content(image_list)
    with open("aurora_report.html", "w") as file:
        file.write(html_content)

    # print("HTML file created successfully!")

## TODO: run AuroraTests, write output to report.txt, mark those that have failures, go through ALL PNGs, generate HTML
