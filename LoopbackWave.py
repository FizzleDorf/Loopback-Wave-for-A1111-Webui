import os
import platform
import numpy as np
from tqdm import trange
import math
import subprocess as sp
import string
import random
from functools import reduce
import re

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
import subprocess


wave_completed_regex = r'@wave_completed\(([\-]?[0-9]*\.?[0-9]+), ?([\-]?[0-9]*\.?[0-9]+)\)'
wave_remaining_regex = r'@wave_remaining\(([\-]?[0-9]*\.?[0-9]+), ?([\-]?[0-9]*\.?[0-9]+)\)'

def run_cmd(cmd):
    cmd = list(map(lambda arg: str(arg), cmd))
    print("Executing %s" % " ".join(cmd))
    popen_params = {"stdout": sp.DEVNULL, "stderr": sp.PIPE, "stdin": sp.DEVNULL}

    if os.name == "nt":
       popen_params["creationflags"] = 0x08000000

    proc = sp.Popen(cmd, **popen_params)
    out, err = proc.communicate()  # proc.wait()
    proc.stderr.close()

    if proc.returncode:
        raise IOError(err.decode("utf8"))

    del proc

def encode_video(input_pattern, starting_number, output_dir, fps, quality, encoding, create_segments, segment_duration, ffmpeg_path):
    two_pass = (encoding == "VP9 (webm)")
    alpha_channel = ("webm" in encoding)
    suffix = "webm" if "webm" in encoding else "mp4"
    output_location = output_dir + f".{suffix}"

    encoding_lib = {
      "VP9 (webm)": "libvpx-vp9",
      "VP8 (webm)": "libvpx",
      "H.264 (mp4)": "libx264",
      "H.265 (mp4)": "libx265",
    }[encoding]

    args = [
        "-framerate", fps,
        "-start_number", int(starting_number),
        "-i", input_pattern, 
        "-c:v", encoding_lib, 
        "-b:v","0", 
        "-crf", quality,
        ]

    if encoding_lib == "libvpx-vp9":
        args += ["-pix_fmt", "yuva420p"]
        
    if(ffmpeg_path == ""):
        ffmpeg_path = "ffmpeg"
        if(platform.system == "Windows"):
            ffmpeg_path += ".exe"

    print("\n\n")
    if two_pass:
        first_pass_args = args + [
            "-pass", "1",
            "-an", 
            "-f", "null",
            os.devnull
        ]

        second_pass_args = args + [
            "-pass", "2",
            output_location
        ]

        print("Running first pass ffmpeg encoding")       

        run_cmd([ffmpeg_path] + first_pass_args)
        print("Running second pass ffmpeg encoding.  This could take awhile...")
        run_cmd([ffmpeg_path] + second_pass_args)
    else:
        print("Running ffmpeg encoding.  This could take awhile...")
        run_cmd([ffmpeg_path] + args + [output_location])

    if(create_segments):
      print("Segmenting video")
      run_cmd([ffmpeg_path] + [
          "-i", output_location,
          "-f", "segment",
          "-segment_time", segment_duration,
          "-vcodec", "copy",
          "-acodec", "copy",
          f"{output_dir}.%d.{suffix}"
      ])
      
def set_weights(match_obj, wave_progress):
  weight_0 = 0
  weight_1 = 0
  if match_obj.group(1) is not None:
    weight_0 = float(match_obj.group(1))
  if match_obj.group(2) is not None:
    weight_1 = float(match_obj.group(2))
    
  max_weight = max(weight_0, weight_1)
  min_weight = min(weight_0, weight_1)
  
  weight_range = max_weight - min_weight
  weight = min_weight + weight_range * wave_progress
  return str(weight)


class Script(scripts.Script):
    def title(self):
        return "Loopback Wave V1.4.1"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        frames = gr.Slider(minimum=1, maximum=2048, step=1, label='Frames', value=100)
        frames_per_wave = gr.Slider(minimum=0, maximum=120, step=1, label='Frames Per Wave', value=20)
        denoising_strength_change_amplitude = gr.Slider(minimum=0, maximum=1, step=0.01, label='Max additional denoise', value=0.6)
        denoising_strength_change_offset = gr.Number(minimum=0, maximum=180, step=1, label='Wave offset (ignore this if you don\'t know what it means)', value=0)
        initial_image_number = gr.Number(minimum=0, label='Initial generated image number', value=0)

        save_prompts = gr.Checkbox(label='Save prompts as text file', value=True)
        prompts = gr.Textbox(label="Prompt Changes", lines=5, value="")

        save_video = gr.Checkbox(label='Save results as video', value=True)
        output_dir = gr.Textbox(label="Video Name", lines=1, value="")
        video_fps = gr.Slider(minimum=1, maximum=120, step=1, label='Frames per second', value=10)
        video_quality = gr.Slider(minimum=0, maximum=60, step=1, label='Video Quality (crf)', value=40)
        video_encoding = gr.Dropdown(label='Video encoding', value="VP9 (webm)", choices=["VP9 (webm)", "VP8 (webm)", "H.265 (mp4)", "H.264 (mp4)"])
        ffmpeg_path = gr.Textbox(label="ffmpeg binary.  Only set this if it fails otherwise.", lines=1, value="")

        segment_video = gr.Checkbox(label='Cut video in to segments', value=True)
        video_segment_duration = gr.Slider(minimum=10, maximum=60, step=1, label='Video Segment Duration (seconds)', value=20)


        return [frames, denoising_strength_change_amplitude, frames_per_wave, denoising_strength_change_offset,initial_image_number, prompts, save_prompts, save_video, output_dir, video_fps, video_quality, video_encoding, ffmpeg_path, segment_video, video_segment_duration]

    def run(self, p, frames, denoising_strength_change_amplitude, frames_per_wave, denoising_strength_change_offset, initial_image_number, prompts: str,save_prompts, save_video, output_dir, video_fps, video_quality, video_encoding, ffmpeg_path, segment_video, video_segment_duration):
        processing.fix_seed(p)
        batch_count = p.n_iter
        p.extra_generation_params = {
            "Max Additional Denoise": denoising_strength_change_amplitude,
            "Frames per wave": frames_per_wave,
            "Wave Offset": denoising_strength_change_offset,
        }

        # We save them ourselves for the sake of ffmpeg
        p.do_not_save_samples = True

        changes_dict = {}


        p.batch_size = 1
        p.n_iter = 1

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        grids = []
        all_images = []
        original_init_image = p.init_images
        state.job_count = frames * batch_count

        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]
        initial_denoising_strength = p.denoising_strength

        if(output_dir==""):
            output_dir = str(p.seed)
        else:
            output_dir = output_dir + "-" + str(p.seed)

        loopback_wave_path = os.path.join(p.outpath_samples, "loopback-wave")
        loopback_wave_images_path = os.path.join(loopback_wave_path, output_dir)

        os.makedirs(loopback_wave_images_path, exist_ok=True)

        p.outpath_samples = loopback_wave_images_path
        
        prompts = prompts.strip()
        
        if save_prompts:
            with open(loopback_wave_images_path + "-prompts.txt", "w") as f:
                generation_settings = [
                  "Generation Settings",
                  f"Total Frames: {frames}",
                  f"Frames Per Wave: {frames_per_wave}",
                  f"Wave Offset: {denoising_strength_change_offset}",
                  f"Base Denoising Strength: {initial_denoising_strength}",
                  f"Max Additional Denoise: {denoising_strength_change_amplitude}",
                  f"Initial Image Number: {initial_image_number}",
                  "",
                  "Video Encoding Settings",
                  f"Save Video: {save_video}"
                ]
                
                if save_video:
                  generation_settings = generation_settings + [
                    f"Framerate: {video_fps}",
                    f"Quality: {video_quality}",
                    f"Encoding: {video_encoding}",
                    f"Create Segmented Video: {segment_video}"
                  ]
                  
                  if segment_video:
                    generation_settings = generation_settings + [f"Segment Duration: {video_segment_duration}"]
                 
                generation_settings = generation_settings + [
                  "",
                  "Prompt Details",
                  "Initial Prompt:" + p.prompt,
                  "",
                  "Negative Prompt:" + p.negative_prompt,
                  "",
                  "Frame change prompts:",
                  prompts
                ]
                  


                f.write('\n'.join(generation_settings))

        if prompts:
            lines = prompts.split("\n")
            for prompt_line in lines:
              params = prompt_line.split("::")
              if len(params) == 2:
                changes_dict[params[0]] = { "prompt": params[1] }
              elif len(params) == 3:
                changes_dict[params[0]] = { "seed": params[1], "prompt": params[2] }
              else:
                raise IOError(f"Invalid input in prompt line: {prompt_line}")
        
        raw_prompt = p.prompt
                
        for n in range(batch_count):
            history = []

            # Reset to original init image at the start of each batch
            p.init_images = original_init_image
            
            seed_state = "adding"
            current_seed = p.seed

            for i in range(frames):
                current_seed = p.seed
                state.job = ""
                
                if str(i) in changes_dict:
                  raw_prompt = changes_dict[str(i)]["prompt"]
                  state.job = "New prompt: %s\n" % raw_prompt
                                    
                  if "seed" in changes_dict[str(i)]:
                    current_seed = changes_dict[str(i)]["seed"]
                    
                    if current_seed.startswith("+"):
                      seed_state = "adding"
                      current_seed = current_seed.strip("+")
                    elif current_seed.startswith("-"):
                      seed_state = "subtracting"
                      current_seed = current_seed.strip("-")
                    else:
                      seed_state = "constant"
                      
                    current_seed = int(current_seed)
                    p.seed = current_seed
                      
                      
                  
                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True

                if opts.img2img_color_correction:
                    p.color_corrections = initial_color_corrections
                    
                    
                wave_progress = float(1)/(float(frames_per_wave - 1))*float(((float(i)%float(frames_per_wave)) + ((float(1)/float(180))*denoising_strength_change_offset)))
                print(wave_progress)
                new_prompt = re.sub(wave_completed_regex, lambda x: set_weights(x, wave_progress), raw_prompt)
                new_prompt = re.sub(wave_remaining_regex, lambda x: set_weights(x, 1 - wave_progress), new_prompt)
                p.prompt = new_prompt
                
                print(new_prompt)

                denoising_strength_change_rate = 180/frames_per_wave

                cos = abs(math.cos(math.radians(i*denoising_strength_change_rate + denoising_strength_change_offset)))
                p.denoising_strength = initial_denoising_strength + denoising_strength_change_amplitude - (cos * denoising_strength_change_amplitude)

                state.job += f"Iteration {i + 1}/{frames}, batch {n + 1}/{batch_count}. Denoising Strength: {p.denoising_strength}"

                processed = processing.process_images(p)

                if initial_seed is None:
                    initial_seed = processed.seed
                    initial_info = processed.info

                init_img = processed.images[0]

                p.init_images = [init_img]
                
                if seed_state == "adding":
                  p.seed = processed.seed + 1
                elif seed_state == "subtracting":
                  p.seed = processed.seed - 1
                  
                image_number = int(initial_image_number + i)
                images.save_image(init_img, p.outpath_samples, "", processed.seed, processed.prompt, forced_filename=str(image_number))

                history.append(init_img)

            grid = images.image_grid(history, rows=1)
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

            grids.append(grid)
            all_images += history

        if opts.return_grid:
            all_images = grids + all_images

        if save_video:
            input_pattern = os.path.join(loopback_wave_images_path, "%d.png")
            encode_video(input_pattern, initial_image_number, loopback_wave_images_path, video_fps, video_quality, video_encoding, segment_video, video_segment_duration, ffmpeg_path)

        processed = Processed(p, all_images, initial_seed, initial_info)

        return processed


