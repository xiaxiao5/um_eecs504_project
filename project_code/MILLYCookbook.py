import fiftyone as fo
import fiftyone.utils.data as foud
import fiftyone.core.metadata as fom
import fiftyone.utils.video as fuv
import os
import glob
import ffmpeg
import json
import multiprocessing
import time
import numpy as np
import eta.core.video as etav
import fiftyone.core.utils as fou
fovi = fou.lazy_import("fiftyone.core.video")
from fiftyone import ViewField as FF
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import fiftyonize_utils as fu
import pose as P

task_spec = {  # a map from recipe to a map from step_no to step text
    0: {
        0: "0. start.",
        1: "1. Place tortilla on cutting board.",
        2: "2. Use a butter knife to scoop nut butter from the jar. Spread nut butter onto tortilla, leaving 1/2 inch uncovered at the edges.",
        3: "3. Clean the knife by wiping with a paper towel.",
        4: "4. Use the knife to scoop jelly from the jar. Spread jelly over the nut butter.",
        5: "5. Clean the knife by wiping with a paper towel.",
        6: "6. Roll the tortilla from one end to the other into a log shape, about 1.5 inches thick. Roll it tight enough to prevent gaps, but not so tight that the filling leaks.",
        7: "7. Secure the rolled tortilla by inserting 5 toothpicks about 1 inch apart.",
        8: "8. Trim the ends of the tortilla roll with the butter knife, leaving 1⁄2 inch margin between the last toothpick and the end of the roll. Discard ends.",
        9: "9. Slide floss under the tortilla, perpendicular to the length of the roll. Place the floss halfway between two toothpicks.",
        10: "10. Cross the two ends of the floss over the top of the tortilla roll. Holding one end of the floss in each hand, pull the floss ends in opposite directions to slice.",
        11: "11. Continue slicing with floss to create 5 pinwheels.",
        12: "12. Place the pinwheels on a plate.",
        13: "13. end.",
    },
    1: {
        0: "0. start.",
        1: "1. Measure 12 ounces of cold water and transfer to a kettle. Boil the water.",
        2: "2. While the water is boiling, assemble the filter cone. Place the dripper on top of a coffee mug.",
        3: "3. Prepare the filter insert by folding the paper filter in half to create a semi-circle, and in half again to create a quarter-circle. Place the paper filter in the dripper and spread open to create a cone",
        4: "4. Weigh the coffee beans and grind until the coffee grounds are the consistency of course sand, about 20 seconds. Transfer the grounds to the filter cone.",
        5: "5. Once the water has boiled, check the temperature. The water should be between 195-205 degrees Fahrenheit or between 91-96 degrees Celsius. If the water is too hot, let it cool briefly.",
        6: "6. Pour a small amount of water in the filter to wet the grounds. Wait for coffee to bloom, about 30 seconds. You will see small bubbles or foam on the coffee grounds during this step.",
        7: "7. Slowly pour the rest of the water over the grounds in a circular motion. Do not overfill beyond the top of the paper filter.",
        8: "8. Let the coffee drain completely into the mug before removing the dripper. Discard the paper filter and coffee grounds.",
        9: "9. end.",
    },
    2: {
        0: "0. start.",
        1: "1. Place the paper cupcake liner inside the mug. Set aside.",
        2: "2. Measure and add the flour, sugar, baking powder, and salt to the mixing bowl.",
        3: "3. Whisk to combine.",
        4: "4. Measure and add the oil, water, and vanilla to the bowl.",
        5: "5. Whisk batter until no lumps remain.",
        6: "6. Pour batter into prepared mug.",
        7: "7. Microwave the mug and batter on high power for 60 seconds.",
        8: "8. Check the cake for doneness; it should be mostly dry and springy at the top. If the cake still looks wet, microwave for an additional 5 seconds.",
        9: "9. Invert the mug to release the cake onto a place. Carefully remove paper liner.",
        10: "10. While the cake is cooling, prepare to pipe the frosting. Scoop a spoonful of chocolate frosting into a zip-top bag and seal, removing as much air as possible",
        11: "11. Use scissors to cut one corner from the bag to create a small opening 1⁄4-inch in diameter.",
        12: "12. Squeeze the frosting through the opening to apply small dollops of frosting to the plate in a circle around the base of the cake.",
        13: "13. end.",
    },
}

# a dict of dict: recipe_no -> step text -> step no
task_spec_text2no = {recipe_no: {step_text: step_no for step_no, step_text in recipe_dict.items()} for recipe_no, recipe_dict in task_spec.items()}  

recipe2no_map = {  # "recipe text" -> "recipe no"
    "A_pin": 0,
    "B_coffee": 1,
    "C_cake": 2,
}

def recipe2no(recipe):
    return recipe2no_map[recipe]
    
def source2env(source, recipe_no):
    environment_map = {  # 'recipe'->'source'->'env'
        0: {
            0: 0,
            1: 0,
            2: 0,
            3: 2,
            4: 2,
            5: 2,
            6: 2,
            7: 0,
            8: 0,
            18: 2,
            19: 1,
            20: 3,
            21: 4,
            22: 5,
            23: 6,
            24: 7,
            25: 8,
            26: 9,
        },
        1: {
            0: 0,
            14: 1,
            10: 2,
            11: 2,
            12: 2,
            6: 2,
            7: 0,
        },
        2: {
            0: 0,
            14: 1,
            15: 2,
            4: 2,
            17: 2,
            6: 2,
            7: 0,
        },
#         100: {  # tourniquet
#             0: 0,
#         },
    }
    return environment_map[recipe_no][source]

all_sensors = ["pv", # RGB
                "rm_vlc_lf", # left front grayscale
                "rm_vlc_rf", # right front grayscale
                "ir", # infrared
                "depth", # depth
                "microphone" # microphone
               ]

splits = ["train", "val"]

def check_mevo_video_for_vis(device, og_path):
    """
    Preprocess those mp4 files that have issues with FO App visualization or header is corrupted.
    
    Specifically, only for mevo data:
    1. check codec. reencode to h.264 if it's mpeg4. mpeg4 is not supported by FO App visualization. Those are videos collected by some versios of milly_capture. h264 is supported by FO App visualization. There are some videos are collected by (multi)MEVO App and Roc recording software are h264(all of them are 720p resolution in this dataset).
    2. check vid size. transform to (1280, 1920) if it's (480, 640) (thry are all mpeg4 as well, reencode as h264). Because videos with shape (480, 640) on the record are those who have corrupted binary files (issue 15). Prob need better way to check corruption since in the future the expected size may change. 
    
    If a new video is generated, new vid use the name pv_clean.mp4, while original vid keep name pv.mp4
    
    return:
        (fo_path, og_path)
    """
    clean_path = og_path
    # return (clean_path, og_path)
    # only mevo camera data has this issue
    if device == "mevo":
        og_vid_probe = ffmpeg.probe(og_path)["streams"][0]
        codec, H, W = og_vid_probe["codec_name"], og_vid_probe["height"], og_vid_probe["width"]
        # if this mevo video has h264 codec, no need to reencode
        if codec == "h264":
            return (clean_path, og_path, codec, H, W)
        else:
            clean_path = og_path.replace("pv.mp4", "pv_clean.mp4")
            # reencode if not done yet
            if (
                not os.path.exists(clean_path)
                    # or ffmpeg.probe(fo_path)["streams"][0]["codec_name"] != "h264"
            ):
                # all mpeg4 codec should be reencoded to h264 for correct visualization
                # # (width, height) expected shape for milly_capture videos are 1080p.
                # TODO: this is not necessarily true for future mevo videos. Need to capture the actual the actual resolution (resolution for latter frames) in other ways
                rightsize = (1920, 1080)
                
                # if the video is already encoded in right shape, just reencode it from mpeg4. Again, this is for correct visualization
                if (W, H) == rightsize:
                    fuv.reencode_video(input_path=og_path, output_path=clean_path, verbose=True)
                # if the video is not the right size reencode with right size
                else:
                    # this video is mpeg4 codec and have wrong size on the record (due to corrupted binary files issue_15)
                    fuv.transform_video(
                        input_path=og_path,
                        output_path=clean_path,
                        size=rightsize,
                        reencode=True,
                        verbose=True,
                    )
            return (clean_path, og_path, codec, H, W)
    # for other sensors, no need to process so just use the same video for visualization
    else:
        return (clean_path, og_path, None, None, None)
     
     
def annotation_file2TDs(sample):
    """
    args:
        sample: fo Video Sample
    return:
        fo.TemporalDetections covers step level annotation
    """
    # export all out of protocal (OOP) annotations to a file. It makes phrasing much harder!
    with open("out_of_protocal_annotations.txt", "a") as oopf:
        #
        total_frame_count = sample.metadata.total_frame_count
        
        # frame rate
        annotation_fps = 30 # people always use pv or rm_vlc_lf/rf for annotation which are 30 fps
        sample_fps = sample.metadata.frame_rate
        
        # load project json file file and video info
        project_file = json.load(open(sample.annotation_filepath, "r"))
        
        seg_dict = project_file["metadata"]
        detections = []        
        for i, seg_name in enumerate(seg_dict.keys()):
            # Out of Protocal(OOP): has meaningless spatial ann
            if len(seg_dict[seg_name]["xy"]) > 0:
                print(
                    f"=={sample.filepath=}==\nthere is a useless spatial annotaion (xy coordinate) detected.\n",
                    file=oopf,
                )
                continue
            
            # get start and end timing of current segment in annotation file
            start, end = seg_dict[seg_name]["z"]
            
            # OOP: If start timing beyond vid length, discard rest of segments. This is caused by the fact that annotation video is not synchronized with other videos. Here we assume the "unsynchronization" means videos are stopped at different timing but recoreded parts are synchronized. But this might not be true.
            # TODO: determin why/how videos are unsynchronized so we can match annotation to videos from different sensors more accurately.
            if start * sample_fps > total_frame_count:
                print(
                    f"=={sample.filepath=}==\nthere is a start timing beyond vid length\n{start * sample_fps=}; {total_frame_count=}; {label=}\n",
                    file=oopf,
                )
                break
            
            # phrase step annotated for this segment
            raw_step_no = seg_dict[seg_name]["av"][
                "1"
            ]  # step_no of current segment; In annotation file, it is the item no. in the 1st attribute of current segment; str.
            step_no = _step_ensure_number(raw_step_no)  # OOP: people don't follow protocol to use prepared annotation template. process it here; int
            if sample.recipe_no == 0 and step_no > 13: # OOP: people don't follow protocol to use prepared annotation template. process it here
                step_no = 13
            label = task_spec[sample.recipe_no][step_no]
            
            
            # make a TemporalDetection for this segment
            TD = fo.TemporalDetection.from_timestamps(
                [start, end], label=label, sample=sample
            )  # just needs total_frame_count and duration in metadata from sample            
            
            detections.append(TD)
            
#             # OOP: end timing beyond vid length
              
        # the last segment should cover the rest of the video
##        detections[-1]["support"][1] = total_frame_count
    return fo.TemporalDetections(detections=detections)
                
def _step_ensure_number(raw_step_no):
    if "." in raw_step_no:
        return int(raw_step_no[0].split(".")[0])
    elif ":" in raw_step_no:
        return int(raw_step_no[0].split(":")[0])
    elif "start" == raw_step_no:
        return 0
    elif "end" == raw_step_no:
        return 1
    else:
        return int(raw_step_no)
    
def annotation_postprocess(naive_TDs, total_frame_count):
    """
    process the TemporalDetections to fit "no-gap & no-overlap" assumption.
    args:
        naive_TDs: fo.TemporalDetections
    return:
        fo.TemporalDetections
    """
    # order annotation segments by start timing of each segment instead of annotation timing instead of annotation timing.
    sorted_naive_detections = sorted(naive_TDs.detections, key=lambda x: x.support[0])
        
    processed_TDs = fo.TemporalDetections()
    # force first segment start with the first frame
    end = 0
    for i, TD in enumerate(sorted_naive_detections):
        
        # fully covered by previous segment or at least totally the same as previous segment (cur_start >= prev_start since detections're sorted)
        if TD.support[0] <= end and TD.support[1] <= end:
            start = TD.support[0]
            
        # partially overlapped w/ previous segment OR having a gap OR seamless to prev segment
        # We fill the gaps with the next step_no. Since usually gaps are caused by a delayed start of the next step instead of an early finish of the previous step.
        else:
            start = end+1
        end = TD.support[1]
                    
        processed_TDs.detections.append(fo.TemporalDetection(support=[start, end], label=TD.label))
        
    # the last segment should cover the rest of the video
    processed_TDs.detections[-1]["support"][1] = total_frame_count
    if processed_TDs.detections[-1]["support"][1] < processed_TDs.detections[-1]["support"][0]:
        processed_TDs.detections[-1]["support"][0] = processed_TDs.detections[-1]["support"][1]
    return processed_TDs
    
def query_step_by_frame_number(sample, frame_number):
    for segment in sample["step"].detections:
        if frame_number >= segment.support[0] and frame_number <= segment.support[1]:
            return segment.label
        else:
            continue
    raise ValueError(f"step annotation for frame number {frame_number} not found in {sample.filepath=}")        
     
class reader_decoded_rm_depth:
    def __init__(self, filename):
        import hl2ss
        chunk=hl2ss.ChunkSize.RM_DEPTH_LONGTHROW
        self._client = hl2ss.raw_reader()
        self._filename = filename
        self._chunk = chunk
        
    def open(self):
        self._client.open(self._filename, self._chunk)
        
    def read(self):
        data = self._client.read()
        if (data is None):
            return None
        data.payload = hl2ss.unpack_rm_depth(data.payload)
        return data
    
    def close(self):
        self._client.close()
         

class ExampleTorchDataset(Dataset):
    def __init__(self, benchmark, split):
        """
        args:
            benchmark: MILLYCookbook.Benchmark object
            split: str. "train" or "val"
        """
        super(ExampleTorchDataset).__init__()
        self.split = split
        self.benchmark = benchmark
    def __len__(self):
        return len(self.benchmark.dataset_narrow_dict[self.split]["anchor_frames"])
    def __getitem__(self, idx):
        """
        In this function, we preprocess one timestamp of data. One timestamp of data covers frames from all user selected sensors as adict.
        here we assume benchmark.random_access_frame_samples return synchronized frames from all sensors. It's its responsibility to make sure this.        
        args:
            idx: int. index of the frame/timestamp.
        return:
            image_dict: a dict of torch.Tensor. {sensor: image}
            label: int. label of the timestamp
            filepath_dict: a dict of str. {sensor: filepath}
        """
        # return a dict of synchronized sensors to frame samples at given timestamp (idx)
        sample_dict = self.benchmark.random_access_frame_samples(self.split, idx)
        
        image_dict = {}
        label = -1
        filepath_dict = {}        
        for sensor, frame_sample in sample_dict.items():            
            # filepath
            filepath_dict[sensor] = frame_sample.filepath
            
            # read and preprocess image
            # [CUSTOMIZE]: user should implement their own image reading and preprocessing function
            image_dict[sensor] = torch.rand(3,224,224)
            
            # label
            label = frame_sample.step.step_no
        
        return image_dict, label, filepath_dict


class MyProcess(multiprocessing.Process):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.kwargs = kwargs
        if self.kwargs["function"] not in ["extract_frames", "build_frames"]:
            self.return_type = self.kwargs["return_type"]
            self.write_fo = self.kwargs["write_fo"]
            if self.kwargs["function"] != "get_pose":
                self.read_type = self.kwargs["read_type"]
                
        self.worker_function = getattr(self, self.kwargs["function"], None)
        assert self.worker_function is not None, f"{self.kwargs['function']} not implemented"
        
        self.dataset = self.kwargs["dataset"]
        self.num_samples = len(self.dataset) # don't touch self.dataset out of lock                
                
        if "target_field" in self.kwargs.keys():
            if self.kwargs["return_type"] == "path":
                self.field_name = self.kwargs["target_field"]+"_path"
            elif self.kwargs["return_type"] == "instance":
                self.field_name = self.kwargs["target_field"]
            else:
                NotImplementedError(f"{self.kwargs['return_type']=} not implemented")
                
        # lock to make sure self.dataset is always valid
        self.lock = multiprocessing.Lock()
        
    def start_and_join(self):
        start_time = time.time()
        print(f"Process {self.kwargs['function']} start")
        
        if self.kwargs["num_workers"] == 0:
            queue = multiprocessing.Queue()
            
            # Create a list of workers
            for sample_index in range(len(self.dataset)):
                try:
                    if self.kwargs['function'] in ['extract_frames', 'build_frames']:                        
                        self.worker_function_wrapper(sample_index, queue, self.worker_function)
                    else:
                        self.worker_function(sample_index, queue)
                except Exception as e:
                    print(f"FAILED {sample_index=} got {e=}")
            print(f"processed")    
            # process and get results
            self.result_dict = {idx: [] for idx in range(len(self.dataset))}
            self.finish_flags = []
            self.running_processes = 0
            while len(self.finish_flags) < len(self.dataset):
                self._queue_get(queue, sleep_time=0)
            print(f"===all process done===")
            
        else:
            self.start()
            self.join()
        print(f"Process {self.kwargs['function']} DONE in {time.time()-start_time:.0f}s")
                
    def run(self):
        """A process is finished only when the its finish flag is captured. In this way, we are not only control cpu workload but also queue's."""
        queue = multiprocessing.Queue()
        
        # process and get results
        num_iters = len(self.dataset)
        self.result_dict = {idx: [] for idx in range(num_iters)}
        self.finish_flags = []
        self.running_processes = 0
        for sample_index in range(len(self.dataset)):
            if self.kwargs['function'] in ['extract_frames', 'build_frames']:                        
                worker = multiprocessing.Process(target=self.worker_function_wrapper, args=(sample_index, queue, self.worker_function))
            else:
                self.worker_function(sample_index, queue)
                worker = multiprocessing.Process(target=self.worker_function, args=(sample_index, queue))            
            worker.start()
            self.running_processes += 1
            while self.running_processes >= self.kwargs["num_workers"]:
                self._queue_get(queue, sleep_time=0)
        
        while len(self.finish_flags) < num_iters:
            self._queue_get(queue, sleep_time=0)
        print(f"===all process done===")
        
    def _queue_get(self, queue, sleep_time=0):
        if not queue.empty():
            result = queue.get()
            if type(result) == tuple:
                self.result_dict[result[0]].append(result[1:])
            elif type(result) == int:
                self.finish_flags.append(result)
                self.running_processes -= 1
                print(f"FINISHED sample index {result}; progress {len(self.finish_flags)}/{self.num_samples}")
            else:
                raise NotADirectoryError("unexpected result type got")
        time.sleep(sleep_time)
        
    def worker_function_wrapper(self, sample_index, queue, worker_function):
        """
        This wrapper is to peel function logic from accessing MongoDB. This can speed up the process by reducing the number of times we access MongoDB.
        """
        try:
            # read fields and sample from disk        
            if "read_field_list" not in self.kwargs.keys():
                read_field_list = []
            else:
                read_field_list = self.kwargs["read_field_list"]
            read_field_dict, sample = self.read_fo_dataset_safe(sample_index, read_field_list)
            
            # function logic
            write_field_dict, sample = self.worker_function(read_field_dict, sample)
            
            # write fields and sample to disk
            if "write_fo" in self.kwargs.keys() and self.kwargs["write_fo"]:
                self.write_fo_dataset_safe(sample_index, write_field_dict, sample)
            queue.put(sample_index)
        except Exception as e:
            print(f"FAILED {sample_index=} got {e=}")
           
    def get_sample_frame_num(self, sample):
        sample_frame_num = 0
        if sample.has_field("support"):
            # clip            
            sample_frame_num = sample.support[1]-sample.support[0]+1
        else:
            # video
            sample_frame_num = sample.metadata.total_frame_count
        return sample_frame_num
            
    def read_fo_dataset_safe(self, sample_index, read_field_list):
        """
        sample_index: index of the sample in the dataset
        field_name_list: list of field names to read
        
        return: list of values of the fields
        """
        self.lock.acquire()
        try:            
            print(f"[safe read] {sample_index=} got lock")
            dataset_one_sample = self.get_one_sample_dataset_view(sample_index)
            # field values
            read_field_dict = {}
            field_value_list = dataset_one_sample.values(read_field_list, unwind=True)
            for field_i in range(len(read_field_list)):
                field_name = read_field_list[field_i] # to test: video level field
                field_value = field_value_list[field_i]
                read_field_dict[field_name] = field_value
                
            # sample
            sample = dataset_one_sample.first()
        except Exception as e:
            print(f"[safe read] when {sample_index=} read self.dataset got error {e=}")
        finally:
            self.lock.release() 
            print(f"[safe read] {sample_index=} released lock")
            
        return read_field_dict, sample
        
    def write_fo_dataset_safe(self, sample_index, result_dict, sample=None):
        """
        resutl_list: a list of values from only one sample to write to the dataset
        """
        self.lock.acquire()
        try:
            print(f"[safe write] {sample_index=} got lock")
            if sample != None:
                sample.save()
                
            dataset_one_sample = self.get_one_sample_dataset_view(sample_index)
            for field_name, field_value in result_dict.items():
                dataset_one_sample.set_values(field_name, [field_value])
        except Exception as e:
            print(f"[safe write] error when write to fo dataset (set_values) {sample_index=} {e=}")
        finally:
            self.lock.release()
            print(f"[safe write] {sample_index=} released lock")
                        
    def get_one_sample_dataset_view(self, sample_index):
        if len(self.dataset) == 0:
            self.dataset.reload()            
            print(f"self.dataset reloaded since it was empty view. {sample_index=}")
        dataset_view = self.dataset.skip(sample_index).limit(1)
        return dataset_view
             
    def extract_frames(self, read_field_dict, sample):
        """
        extract frames for this sample to target filepath in target frame info dir on disk based on arguments of to_frames
        This function won't repeatedly extract frames even when self.dataset is a clip dataset. But please still try to use unoverlapped clips dataset to avoid going through begining of the video repeatedly.
        # related functions: fovi.make_frames_dataset, fouv.transform_video
        """
        if sample.sensor.name in ["pv", "rm_vlc_lf", "rm_vlc_rf"]:
            video_path = sample.filepath
            if "frames_patt" not in self.kwargs or self.kwargs["frames_patt"]==None:
                frames_patt = "%010d.jpg"
            else:
                frames_patt = self.kwargs["frames_patt"]
            images_patt = os.path.join(os.path.splitext(sample.filepath)[0], frames_patt)
            
            # check extract completion; core function is `fovi._get_non_existent_frame_numbers`
            doc_frame_numbers, sample_frame_numbers = fovi._parse_video_frames(
                video_path=video_path,
                sample_frames=False if "sample_frames" not in self.kwargs else self.kwargs["sample_frames"],
                images_patt=images_patt,
                support=None if "support" not in sample.field_names else sample.support,
                total_frame_count=sample.metadata.total_frame_count,
                frame_rate=sample.metadata.frame_rate, # actual frame rate of the video
                frame_ids_map={},
                force_sample=False if "force_sample" not in self.kwargs else self.kwargs["force_sample"],
                sparse=False if not "sparse" in self.kwargs else self.kwargs["sparse"],
                fps=None if not "fps" in self.kwargs else self.kwargs["fps"], # fps of the frames to extract
                max_fps=None if not "max_fps" in self.kwargs else self.kwargs["max_fps"],
                verbose=False if not "verbose" in self.kwargs else self.kwargs["verbose"],
            )            
            # extract if not completely extracted
            if sample_frame_numbers == None or len(sample_frame_numbers) > 0:
                target_frame_nos = sample_frame_numbers if sample_frame_numbers != None else doc_frame_numbers
                print(f"extracting {len(target_frame_nos)} frames for {sample.filepath=}")
                etav.sample_select_frames(
                    video_path=video_path,
                    frames=target_frame_nos,
                    output_patt=images_patt,
                    fast=True
                )
            else:
                print(f"{sample.filepath} already extracted")
                
        elif sample.sensor.name == "depth":
            bin_path = sample.bin_filepath
            output_dir_depth = bin_path.replace(".bin", f"_depth")
            output_dir_ir = bin_path.replace(".bin", "_ir")
            if "frames_patt" not in self.kwargs or self.kwargs["frames_patt"]==None:
                frames_patt = "%010d.npy"
            else:
                frames_patt = self.kwargs["frames_patt"]
            images_patt_depth = os.path.join(os.path.dirname(sample.filepath), "rm_depth_lt_depth", frames_patt)
            images_patt_ir = os.path.join(os.path.dirname(sample.filepath), "rm_depth_lt_ir", frames_patt)
            
            # check frmaes to be extracted
            # TODO: the sample.metadata.total_frame_count is not accurate for depth/ir sensor (inferenced by #pv frames and frame rate). Thus, there will be a chance that the target_frame_numbers is empty even when there are frames to be extracted OR the target_frame_numbers is not empty but there are no frames to be extracted (but the extracting code will still be running). But to get accurate #frames, we need to parse the bin file (foobar: the reason why we want to get frame count is to avoid parsing the bin file for extracting frames, but now we have to parse the bin file to get frame count...)
            target_frame_numbers_depth = fovi._get_non_existent_frame_numbers(images_patt_depth, list(range(1, sample.metadata.total_frame_count+1)))
            target_frame_numbers_ir = fovi._get_non_existent_frame_numbers(images_patt_ir, list(range(1, sample.metadata.total_frame_count+1)))
            target_frame_numbers = list(set(target_frame_numbers_depth) | set(target_frame_numbers_ir))
            print(f"{len(target_frame_numbers)}/{sample.metadata.total_frame_count} frames to be extracted for each depth/ir from {bin_path=}")
            
            # extract frames as needed
            if len(target_frame_numbers) > 0:
                os.makedirs(output_dir_depth, exist_ok=True)
                os.makedirs(output_dir_ir, exist_ok=True)
                reader = reader_decoded_rm_depth(bin_path)
                try:
                    reader.open()
                    frame_no = 1
                    while True:
                        data = None # incase reader in error and data keeps the same value all the time
                        data = reader.read()   
                        if not data:
                            break
                        else:
                            if frame_no not in target_frame_numbers:
                                pass
                            else:
                                # write np; (H, W, 1); frame number start from 1 not 0
                                np.save(os.path.join(output_dir_depth, f"{frame_no:010d}"), data.payload.depth)
                                np.save(os.path.join(output_dir_ir, f"{frame_no:010d}"), data.payload.ab)
                            frame_no += 1
                except Exception as e:
                    print(f"Got error when extracting frames for {bin_path=} {e} ")
                finally:
                    reader.close()                    
                sample["infered_total_frame_count"] = sample.metadata.total_frame_count
                sample.metadata.total_frame_count = frame_no - 1
                sample.save()
            else:
                print(f"{sample.filepath} already extracted")
        elif sample.sensor.name == "ir":
            print(f"extracting frames for sensor={sample.sensor.name} is implemented to perform with sensor=depth (since they all from the same binary file")
        elif sample.sensor.name == "microphone":
            print(f"no frames to be extracted for sensor={sample.sensor.name}")
        else:            
                
            raise NotImplementedError(f"extracting frames from sensor {sample.sensor.name} not supported")
        
        return {}, sample
    
    def build_frames(self, read_field_dict, sample):
        if sample.sensor.name in ["pv", "rm_vlc_lf", "rm_vlc_rf"]:
            ext = ".jpg"
        elif sample.sensor.name in ["depth", "ir"]:
            ext = ".npy"
        else:
            return {}, None
        total_frame_count = self.get_sample_frame_num(sample)
        frame_par_dir = os.path.splitext(sample.filepath)[0] # the video path w/o postfix
        # initialize Frame instances
        
        sample.frames = {}
        for frame_no in range(1, total_frame_count+1):
            filepath=os.path.join(frame_par_dir, f"{frame_no:010d}"+ext)
            step_text = query_step_by_frame_number(sample, frame_no)
            step_no = task_spec_text2no[sample.recipe_no][step_text]
            step=fo.Classification(label=step_text, step_no=step_no)
            sample.frames[frame_no] = fo.Frame(filepath=filepath, step=step, sensor_name=sample.sensor.name, dish_path=sample.dish_path, recipe_no=sample.recipe_no, frame_rate=sample.metadata.frame_rate)
        
        result_dict = {}
        return result_dict, sample
    
    def get_pose(self, sample_index, queue):
        self.lock.acquire()
        try:
            dataset_one_sample = self.check_dataset_view(sample_index, self.dataset.skip(sample_index).limit(1))
            filepath_list, frame_info_dir_list = dataset_one_sample.values(["frames.filepath", "frames.frame_info_dir"], unwind=True)
        except Exception as e:
            print(f"when {sample_index=} working with self.dataset got error {e=}")
        finally:
            self.lock.release()
            
        print(f"start {len(filepath_list)=} {sample_index=} / {self.num_samples=}")
        pose_list_one_vid = []        
        pose_detector = P.get_pose_detector()
        for frame_index, (filepath, frame_info_dir) in enumerate(zip(filepath_list, frame_info_dir_list)):
            pose_list_one_vid.append(P.get_pose(filepath, frame_info_dir, pose_detector, return_type=self.return_type))
        if self.write_fo:
            self.lock.acquire()
            try:
                dataset_one_sample = self.check_dataset_view(sample_index, dataset_one_sample)
                dataset_one_sample.set_values(self.field_name, [pose_list_one_vid])
            except Exception as e:
                print(f"error when set_values {sample_index=} {e=}")
            finally:
                self.lock.release()
        pose_detector.close()
        queue.put(sample_index)
        print(f"done {sample_index=}")
        # {self.dataset.summary().split('View stages:')[-1]}
        
    def check_completion(self, sample_index, queue):
        print(f"{sample_index=} acquiring lock")
        self.lock.acquire()
        try:
            print(f"{sample_index=} got lock")
            dataset_one_sample = self.check_dataset_view(sample_index, self.dataset.skip(sample_index).limit(1))
#             self.check_dataset(sample_index)
#             dataset_one_sample = self.dataset.skip(sample_index).limit(1)
            filepath_list, frame_info_dir_list, frame_pose_list, frame_number_list = dataset_one_sample.values(["frames.filepath", "frames.frame_info_dir", "frames.pose", "frames.frame_number"], unwind=True)
            sample = dataset_one_sample.first()
        except Exception as e:
            print(f"when {sample_index=} working with self.dataset got error {e=}")
        finally:
            print(f"{sample_index=} released lock")
            self.lock.release()
            
        print(f"start {len(filepath_list)=} {sample_index=} / {self.num_samples=}")
        if self.read_type=="path":
            # frame_pose_list = [P.read_Keypoints(P.get_local_path(frame_info_dir, P.pose_file_name)) for frame_info_dir in frame_info_dir_list]
            frame_pose_list = [] 
            for ffi, frame_info_dir in enumerate(frame_info_dir_list):
                try:
                    frame_pose_list.append(P.read_Keypoints(P.get_local_path(frame_info_dir, P.pose_file_name)))
                except Exception as e:
                    pose_detector = P.get_pose_detector()                    
                    frame_pose_list.append(P.get_pose(filepath_list[ffi], frame_info_dir_list[ffi], pose_detector, force_generate=True, return_type="instance"))
                    pose_detector.close()
                    print(f"reading local pose error {e=}. redetected")
        elif self.read_type=="instance":
            # frames.pose has the instance
            pass
        
        shape = (sample.metadata.frame_height, sample.metadata.frame_width)
        return_list_one_sample = []
        
        # check first
        # return_list_one_sample.append(P.check_first_frame(frame_pose_list[0]))
        for filepath_idx in range(len(filepath_list)-1):
            # first frame
            if filepath_idx == 0:
                idx_0_check_return = P.check_first_frame(frame_pose_list[0], frame_info_dir_list[filepath_idx], self.return_type)
                if idx_0_check_return != None: # updated
                    if self.return_type=="path":
                        frame_pose_list[filepath_idx] = P.read_Keypoints(P.get_local_path(frame_info_dir_list[filepath_idx], P.pose_file_name))
                    elif self.return_type=="instance":
                        frame_pose_list[filepath_idx] = idx_0_check_return
                    return_list_one_sample.append(idx_0_check_return)
                elif self.return_type=="instance":
                    return_list_one_sample.append(frame_pose_list[filepath_idx+1])
                elif self.return_type=="path":
                    return_list_one_sample.append(filepath_list[filepath_idx+1])
            # rest frames
            check_return = P.check_rest_frames(Keypoints_down=frame_pose_list[filepath_idx+1],
                                                     filepath_up_list=[filepath_list[filepath_idx]],
                                                     filepath_down=filepath_list[filepath_idx+1],
                                                     frame_info_dir_down=frame_info_dir_list[filepath_idx+1],
                                                     Keypoints_up_list=[frame_pose_list[filepath_idx]],
                                                     shape=shape,
                                                     registeror = self.registeror,
                                                     return_type=self.return_type)
            
            if check_return != None:
                if self.return_type=="path":
                    frame_pose_list[filepath_idx+1] = P.read_Keypoints(P.get_local_path(frame_info_dir_list[filepath_idx+1], P.pose_file_name))
                elif self.return_type=="instance":
                    frame_pose_list[filepath_idx+1] = check_return
                return_list_one_sample.append(check_return)
            elif self.return_type=="instance":
                return_list_one_sample.append(frame_pose_list[filepath_idx+1])
            elif self.return_type=="path":
                return_list_one_sample.append(filepath_list[filepath_idx+1])
        if self.write_fo:
            self.lock.acquire()
            try:
                dataset_one_sample = self.check_dataset_view(sample_index, dataset_one_sample)
                dataset_one_sample.set_values(self.field_name, [return_list_one_sample])
            except Exception as e:
                print(f"error when set_values {sample_index=} {e=}")
            finally:
                self.lock.release()
        queue.put(sample_index)
        print(f"done {sample_index=}")
        
    def make_forecasting_gt(self, sample_index, queue):
        print(f"{sample_index=} acquiring lock")
        self.lock.acquire()
        try:
            print(f"{sample_index=} got lock")
            dataset_one_sample = self.check_dataset_view(sample_index, self.dataset.skip(sample_index).limit(1))
            filepath_list, frame_info_dir_list, frame_Keypoints_list = dataset_one_sample.values(["frames.filepath", "frames.frame_info_dir", "frames.pose"], unwind=True)
            sample = dataset_one_sample.first()
        except Exception as e:
            print(f"when {sample_index=} working with self.dataset got error {e=}")
        finally:
            print(f"{sample_index=} released lock")
            self.lock.release()
            
        print(f"start {len(filepath_list)=} {sample_index=} / {self.num_samples=}")
        if self.read_type=="path":
            # load pose Keypoints from disk
            frame_Keypoints_list = [] 
            for ffi, frame_info_dir in enumerate(frame_info_dir_list):
                try:
                    frame_Keypoints_list.append(P.read_Keypoints(P.get_local_path(frame_info_dir, P.pose_file_name)))
                except Exception as e:
                    pose_detector = P.get_pose_detector()
                    frame_Keypoints_list.append(P.get_pose(filepath_list[ffi], frame_info_dir_list[ffi], pose_detector, force_generate=True, return_type="instance"))
                    pose_detector.close()
                    print(f"reading local pose error {e=}. redetected")
        elif self.read_type=="instance":
            # frames.pose has the instance loaded in memory
            pass
        
        shape = (sample.metadata.frame_height, sample.metadata.frame_width)
        result_list_one_sample = []
        
        for filepath_idx in range(len(filepath_list)):
            if filepath_idx % (len(filepath_list)//5) == 0:
                print(f"frame progress: {sample_index=} {filepath_idx}/{len(filepath_list)}")
            filepath_list_loop = filepath_list[filepath_idx:filepath_idx+self.forecasting_steps+1]
            Keypoints_list_loop = frame_Keypoints_list[filepath_idx:filepath_idx+self.forecasting_steps+1]
            if len(filepath_list_loop) < (self.forecasting_steps+1):
                filepath_list_loop += [filepath_list_loop[-1]] * ((self.forecasting_steps+1)-len(filepath_list_loop)) # note they share the same data in memory. One changed, all changed
                Keypoints_list_loop += [Keypoints_list_loop[-1]] * ((self.forecasting_steps+1)-len(Keypoints_list_loop)) # note they share the same data in memory
            try:
                forecasting_return = P.make_forecasting_groundtruth(
                    filepath_list=filepath_list_loop,
                    Keypoints_list=Keypoints_list_loop,
                    shape=shape,
                    registeror=self.registeror,
                    frame_info_dir = frame_info_dir_list[filepath_idx],
                    force_generate=self.kwargs["force_generate"],
                    return_type=self.return_type,
                    skip_generate=self.kwargs["skip_generate"]
                )
                result_list_one_sample.append(forecasting_return)
            except Exception as e:
                print(e)
            
        if self.write_fo:
            self.lock.acquire()
            try:
                dataset_one_sample = self.check_dataset_view(sample_index, dataset_one_sample)
                dataset_one_sample.set_values(self.field_name, [result_list_one_sample])
            except Exception as e:
                print(f"error when set_values {sample_index=} {e=}")
            finally:
                self.lock.release()
                
#         dataset_one_sample.set_values(self.field_name, [result_list_one_sample], skip_none=True)
        queue.put(sample_index)
        print(f"done {sample_index=}")

    def check_dataset_view(self, sample_index, dataset_view):
        if len(dataset_view) == 0:
            dataset_view.reload()
            print(f"dataset_view reloaded since it was empty view. {sample_index=}")
        return dataset_view             
    
class MILLYCookbookImporter(foud.GroupDatasetImporter):
    """Custom importer for grouped datasets.
    
    Args:
        dataset_dir (None): the dataset directory. This may be optional for
            some importers
        shuffle (False): whether to randomly shuffle the order in which the
            samples are imported
        seed (None): a random seed to use when shuffling
        max_samples (None): a maximum number of samples to import. By default,
            all samples are imported
        **kwargs: additional keyword arguments for your importer
    """
    
    def __init__(
        self,
        dataset_dir=None,
        shuffle=False,
        seed=None,
        max_samples=None,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples
        )
        self.kwargs = kwargs
        self.dataset_dir = dataset_dir
        video_folder_pattern = os.path.join(
            self.dataset_dir, "*", "*", "[!_]*", "*"
        )
        self.dish_folder_list = glob.glob(video_folder_pattern)
        self.dish_folder_list.sort()
        
        # if in debug mode, only import 3 video for each recipe. This for speed concern (espectially when writing stuff to disk e.g., reencode videos for visualization, extract frames...).
        if self.kwargs["debug_mode"]:
            k = 2
            short_dish_dict = {r: []for r in ["A_pin", "B_coffee", "C_cake"]}
            for dish_folder_path in self.dish_folder_list:
                for recipe_name in short_dish_dict.keys():
                    if recipe_name in dish_folder_path and len(short_dish_dict[recipe_name])<k:
                        short_dish_dict[recipe_name].append(dish_folder_path)
                        break
            short_dish_list = []
            for dish_folder_path in short_dish_dict.values():
                short_dish_list += dish_folder_path
                
            self.dish_folder_list = short_dish_list
        self.group_no = 0
        
        self.group_field_var = "sensor"
        
        # import all sensors. it's fast anyway. one can do filtering later on FO.Dataset
        self.sensor_list = all_sensors
        
        self.sample_list = self.get_sample_list()
        
    def get_sample_list(self):
        sample_list = []
        for dish_folder in self.dish_folder_list:
            recipe_no = recipe2no(dish_folder.split("/")[-4])
            for sensor in self.sensor_list:
                if recipe_no == 0:  # pinwheels
                    sample_list.append(
                        os.path.join(dish_folder, sensor + ".mp4")
                    )
                    break
                else:
                    sample_list.append(os.path.join(dish_folder, sensor + ".mp4"))
        sample_list.sort()
        return sample_list
        
    def __len__(self):
        """The total number of samples that will be imported across all group
        slices.
        """
        return len(self.sample_list)
    
    def __next__(self):
        """Returns information about the next group in the dataset.
        
        Returns:
            a dict mapping slice names to :class:`fiftyone.core.sample.Sample`
            instances
        
        Raises:
            StopIteration: if there are no more groups to import
        """
        # Implement loading the next group in your dataset here
        try:
            dish_folder = self.dish_folder_list[self.group_no]
        except Exception as e:
            raise StopIteration
        
        # the dict to be returned. sensor_name -> sample
        group = {} 
        
        # prep paths
        annotation_path = glob.glob(os.path.join(dish_folder, "*.json"))[0]
        # dish (group) lev info
        recipe, device, source, = dish_folder.split(
            "/"
        )[-4:-1]
        source = int(source)
        recipe_no = recipe2no(recipe)
        env = source2env(source, recipe_no)
        
        for sensor_name in self.sensor_list:
            # vid path
            sample_path = os.path.join(dish_folder, sensor_name + ".mp4")
            
            if recipe_no == 0 and sensor_name != "pv":
                continue        
                
            # vid level info
            sample = fo.Sample(filepath=sample_path)
            sample["recipe"] = fo.Classification(label=recipe)
            sample["recipe_no"] = recipe_no
            sample["device"] = device
            sample["source"] = source
            sample["environment"] = env
            sample["annotation_filepath"] = annotation_path
            sample["dish_path"] = dish_folder
            sample["dish_path_rel"] = dish_folder.split(self.dataset_dir)[-1][1:] # relative path w/o the the first char "/"
            
            # don't compute metadata for microphone. not supported by fo and not necessary for now
            if sensor_name != "microphone":
                # metadata                
                # for ab/ir, we can't directly compute the metadata from bin file
                if sensor_name in ["ir", "depth"]:
                    sample["bin_filepath"] = os.path.join(os.path.dirname(sample_path), "rm_depth_lt.bin")
                    irdp_fps = 5
                    # count frames for ab/ir videos. TODO: It'd be better to load the bin file to count
                    total_frame_count = group["pv"].metadata.total_frame_count // (group["pv"].metadata.frame_rate // irdp_fps)
                    
                    sample.metadata = fom.VideoMetadata(
                        frame_rate=irdp_fps,
                        frame_width=320,
                        frame_height=280,
                        total_frame_count=total_frame_count,
                        duration=total_frame_count / irdp_fps, # notice that for the same dish, duration for all sensors are the same.
                    )
                # other sensors are normal mp4 videos 
                else:
                    sample.compute_metadata(skip_failures=False)
                    
            group[sensor_name] = sample
        self.group_no += 1
        return group
    
    @property
    def has_dataset_info(self):
        """Whether this importer produces a dataset info dictionary."""
        # Return True or False here # TODO
        return False if self.get_dataset_info() == None else True

    @property
    def has_sample_field_schema(self):
        """Whether this importer produces a sample field schema."""
        # Return True or False here
        return False if self.get_sample_field_schema() == None else True

    @property
    def group_field(self):
        """The name of the group field to populate on each sample."""
        # This is the default, but you can customize if desired
        return self.group_field_var

    def setup(self):
        """Performs any necessary setup before importing the first sample in
        the dataset.

        This method is called when the importer's context manager interface is
        entered, :func:`DatasetImporter.__enter__`.
        """
        # Your custom setup here
        pass

    def get_dataset_info(self):
        """Returns the dataset info for the dataset.

        By convention, this method should be called after all samples in the
        dataset have been imported.

        Returns:
            a dict of dataset info
        """
        # Return a dict of dataset info, if supported by your importer
        pass
        

    def get_sample_field_schema(self):
        """Returns a dictionary describing the field schema of the samples
        loaded by this importer.

        The returned dictionary should map field names to to string
        representations of :class:`fiftyone.core.fields.Field` instances
        generated by ``str(field)``.

        Returns:
            a dict
        """
        # Return the sample schema here, if known

        pass

    def close(self, *args):
        """Performs any necessary actions after the last sample has been
        imported.

        This method is called when the importer's context manager interface is
        exited, :func:`DatasetImporter.__exit__`.

        Args:
            *args: the arguments to :func:`DatasetImporter.__exit__`
        """
        # Your custom code here to complete the import
        pass
                
        
class Benchmark():
    def __init__(self, dataset_dir, k_folds=3, sensors=['pv'], seed=51, force_reload=False, persistent=True, debug_mode=False, extract_frames=False, to_frames=False, visualize=True):
        """
        a =Benchmark= object for k-fold cross validation on frame-based step recognition task. It contains
        1. =build_fo_dataset=, a function to prepare Fiftyone Dataset on full dataset
        2. =narrow_down_dataset=, a function to narrow down the Fiftyone Dataset to specific scope as needed
        3. =get_dataloaders=, a function to return PyTorch Dataloader for narrowed Fiftyone Datset veiw
        4. =random_access_frame_samples=, a function for random access to frame-level =Fiftyone.Sample=
        5. =load_predictions=, a function to load predictions to Fiftyone Dataset
        6. =evaluate=, a function to evaluate prediction        
        
        Args:
            dataset_dir (str): path to the dataset directory on your disk (e.g. /your/dataset/folder/path/MILLYCookbook_media_v007)
            k_folds (int): number of folds for cross validation. Has to be <= 10. randomness is controled by seed
            sensors (list): list of sensors to be used *for frame level index building only (e.g., extracting frames and Frame sample building*. Default is ['pv']. Video level index building always covers all sensors.
            seed (int): random seed. Default is 51.
            force_reload (bool): whether to force reload the Fiftyone Dataset Index. Default is False.
            persistent (bool): whether to keep the Fiftyone Dataset Index on disk. Default is True. Typically, you want to keep it on disk for faster loading especially for this big dataset. However, if you are debugging this class, it's better to set it to False since you will load the dataset multiple times.
            debug_mode (bool): whether to run in debug mode. Default is False. If True, it will only load a small subset of the dataset for faster debugging. Specifically, it will load only 2 dishes per recipe and do 2-fold split for each recipe.
            extract_frames (bool): whether to extract frames from videos. Default is False. If True, it will extract frames from videos and save them to disk. WARING: this is ultra time consuming even though we implemented multiprocessing for it. It will take 1-2 days to extract frames from the whole dataset. It's better to set it to False and down the pre-extracted frames. It'd be even better if you have efficient way to rando access frames from videos which means you dont' have to extract frames at all.
            to_frames (bool): whether to build frames level samples. Default is True. If True, it will build frame-level Fiftyone Index (Frame samples) which is required for our training example. Depends on how many cores your machine has, it will take from 30 mins to 2 hours to build the index.
                We found it's easier and efficient to random access frames with this frame level index built. But feel to implement your own way to random access frames from video level samples.
            visualize (bool): whether to visualize the dataset. Default is True. If False, it will visualize the dataset in Fiftyone App.
        """
        assert k_folds<=10, "only supprot k_folds <= 10"
        
        self.dataset_dir = dataset_dir
        assert os.path.exists(self.dataset_dir), f"dataset dir {self.dataset_dir} doesn't exist"
        self.k_folds = k_folds
        self.sensors = sensors
        self.seed = seed
        self.force_reload = force_reload
        self.persistent = persistent
        self.debug_mode = debug_mode
        self.extract_frames = extract_frames
        self.to_frames = to_frames
        self.visualize = visualize
        self.session = None
        
        # build dataset
        self.dataset = None
        self.build_fo_dataset()
        
        self.narrowed_down = False
        
        # dict to store accuracy for different folders, splits
        self.accuracy_dict = {fold_i: {s: {} for s in splits} for fold_i in range(self.k_folds)}
    def set_default_dataset_narrow(self):
        self.narrow_down_dataset(recipe_no=0, fold_no=0, user_sensors=[s for s in all_sensors if s!="microphone"], anchor_sensor="pv")
        
    def build_fo_dataset(self):
        # dataset name
        dataset_name = f"MILLYCookbook_FO_v014"
        
        # check if dataset exists
        if not fo.dataset_exists(dataset_name) or self.force_reload:
            # delete the dataset if it exists
            if fo.dataset_exists(dataset_name):
                input(f"==DO YOU REALLY WANT TO DELETE {dataset_name}?==\n Enter to confirm while Ctrl+c Ctrl+c to cancel\n")
                fo.delete_dataset(dataset_name)
                
            # initialize dataset and load videos to dataset
            print(f"==loading videos to FiftyOne==")
            self.dataset = fo.Dataset.from_importer(MILLYCookbookImporter(dataset_dir=self.dataset_dir, seed=51, debug_mode=self.debug_mode), name=dataset_name)
            self.dataset.persistent = self.persistent
            self.dataset.save()
         
            # visualize
            if self.visualize:
                self.session = fo.launch_app(dataset=self.dataset)
                
            # split dataset into k folds for each recipe at dish level
            print(f"======tagging split info to FO Dataset=====")
            ratio_val = 1/self.k_folds
            # Explicitly save k folds splits for each recipe. relative dish folder path is the path dropped dataset_dir prefix.
            # e.g., {recipe name: {k fold: {train: [relative dish folder path], val: [relative dish folder path]}}}
            self.dataset.info["split"] = {r: {f"fold_{k_i:02d}": {} for k_i in range(self.k_folds)} for r in self.dataset.distinct("recipe.label")}
            # save split info as tags along the way
            
            for recipe_label in self.dataset.distinct("recipe.label"):
                recipe_dataset = self.dataset.match(FF("recipe.label")==recipe_label)
                all_dishes_rela_path = recipe_dataset.shuffle(seed=self.seed).values("dish_path_rel")
                fold_length = int(ratio_val * len(all_dishes_rela_path))
                for k_i in range(self.k_folds):
                    start = k_i * fold_length
                    end = (k_i+1) * fold_length
                    val_dishes = all_dishes_rela_path[start:end]
                    train_dishes = all_dishes_rela_path[:start] + all_dishes_rela_path[end:]
                    self.dataset.info["split"][recipe_label][f"fold_{k_i:02d}"] = {"train": train_dishes, "val": val_dishes}
                    # tag all video samples in this dish as val or train for this fold
                    recipe_dataset.select_group_slices(all_sensors).match(FF("dish_path_rel").is_in(val_dishes)).tag_samples(f"fold_{k_i:02d}_val")
                    recipe_dataset.select_group_slices(all_sensors).match(FF("dish_path_rel").is_in(train_dishes)).tag_samples(f"fold_{k_i:02d}_train")
                    recipe_dataset.save()
                    
            # load step annotations video level samples
            print(f"==loading annotations to video samples==")
            self.load_video_ann()
            
            # extract frames with multiprocessing. It takes 7828s on on zoom (12-cores) for 5 dishes / recipes (55 videos to be extracted in total)
            if self.extract_frames:
                print("================EXTRACTING FRAMES with multiprocessing (one time work if persistent=True)=================")
                fargs = {"sample_frames": True,
                         "frames_patt": None,
                         "verbose": True,
                         "force_sample": False,
                         "function": "extract_frames",
                         "num_workers": os.cpu_count(),
                         "dataset": self.dataset.select_group_slices([sensor for sensor in self.sensors if sensor != "microphone"]) # self.dataset.select_group_slices("depth")
                         }
                start_time = time.time()
                process = MyProcess(**fargs)
                process.start_and_join()
                print(f"===============EXTRACTING FRAMES DONE in {time.time()-start_time:.0f}s==============")                            
                        
            # if load step annotations to Frames. It takes 280s on on zoom (12-cores) for 5 dishes / recipes (55 videos to be extracted in total). 130s for 2 dishes/recipes
            if self.to_frames:
                # initialize fields: need to explicity initialize frame.filepath fields (add it to Dataset Doc) or there will be error in sample.save()
                init_field_dict = {"frames.filepath": "x", "frames.step": fo.Classification(), "frames.sensor_name": "x", "frames.dish_path": "x", "frames.recipe_no": -1, "frames.frame_rate": -1}
                for field_name, field_example in init_field_dict.items():
                    vs = self.dataset.first()
                    vs.frames[1][field_name.split("frames.")[-1]] = field_example
                    vs.save()
                # use mp to initialize Frame instances and load step annotations to Frame instances
                print("============building Frame samples with multiprocessing (one time work)==========")
                fargs = {"function": "build_frames",
                         "num_workers": os.cpu_count(),
                         "dataset": self.dataset.select_group_slices([sensor for sensor in self.sensors if sensor != "microphone"]),
                         "write_fo": True,
                         }
                start_time = time.time()
                process = MyProcess(**fargs)
                process.start_and_join()
                print(f"==============DONE in {time.time()-start_time:.0f}s==============")
                                
        else:
            self.dataset = fo.load_dataset(dataset_name)
            
        return 
    
    def get_fo_dataset(self):
        return self.dataset
    
    def narrow_down_dataset(self, recipe_no, fold_no, user_sensors, anchor_sensor="pv"):
        """
        narrow down to a specific recipe, fold, split, and sensors.
        args:
            recipe_no: int. check global variable "recipe2no_map"
            fold_no: int. fold_no>=0 and fold_no<self.k_folds. self.k_folds is defined in __init__
            split_name: str. "train" or "val"
            sensor_names: list of str. check global variable "all_sensors". Note that "microphone" doesn't have frame level annotations
            anchor_sensor: str. the sensor used to anchor different sensors. For example, if anchor_sensor="pv", then the frame number of pv is used to inference frame numbers for other sensors.
        return:
            fo.Dataset
        """        
        assert "microphone" not in user_sensors and "micropone" != anchor_sensor, "microphone doesn't have frame level annotations"
        if recipe_no != 0 and (len(user_sensors) > 1 or "pv" not in user_sensors or anchor_sensor != "pv"):
            raise ValueError("for pin recipe, only pv sensor should be used")
        if self.narrowed_down:
            print(f"your previous scope of dataset:")
            self.print_scope_info()
        else:
            print(f"this is the first time you narrow down the dataset")
        self.recipe_no = recipe_no
        self.fold_no = fold_no
        self.user_sensors = user_sensors
        self.anchor_sensor = anchor_sensor
        self.dataset_narrow_dict = {s: {} for s in splits} 
        for split_name in splits:
            split_tag = self.get_split_tag(fold_no, split_name)
            print(f"===builiding narrowed Frames view===")
            self.dataset_narrow_dict[split_name]["anchor_frames"] = self.dataset.match(FF("recipe_no") == recipe_no).match_tags(split_tag).select_group_slices(anchor_sensor).to_frames()
            self.dataset_narrow_dict[split_name]["all_vids"] = self.dataset.match(FF("recipe_no") == recipe_no).match_tags(split_tag).select_group_slices(user_sensors)
            print(f"===builiding narrowed Frames view===")
            self.dataset_narrow_dict[split_name]["all_frames"] = self.dataset_narrow_dict[split_name]["all_vids"].to_frames()
        self.narrowed_down = True
        print(f"your current scope of dataset:")
        self.print_scope_info()
    def get_split_tag(self, fold_no, split_name):
        return f"fold_{fold_no:02d}_{split_name}"
    
    def print_scope_info(self):
        print(f"{self.recipe_no=}\n{self.fold_no=}\n{self.user_sensors=}\n{self.anchor_sensor=}")
    
    def random_access_frame_samples(self, split, frame_index):
        """
        random access frame samples from a split given an frame index. This function is usualy invoked in a the __getitem__ function of a Pytorch Dataset.
        args:
            split: str. "train" or "val"
            index: int. index of the frame sample
        return:
            frame_samples: dict. key: sensor name, value: Frame sample
        """
        assert self.narrowed_down, "you have to narrow down dataset to your scope first. check function 'Benchmark.narrow_down_dataset'"
        assert frame_index < len(self.dataset_narrow_dict[split]["anchor_frames"]), f"index {frame_index} is out of range of {self.dataset_narrow_dict[split]['anchor_frames']}"
        # get the frame sample from each sensor
        frame_samples = {self.anchor_sensor: self.dataset_narrow_dict[split]["anchor_frames"].skip(frame_index).first()}
        anchor_frame_rate = int(frame_samples[self.anchor_sensor].frame_rate)
        for sensor in self.user_sensors:
            if sensor == self.anchor_sensor:
                continue
            # locate the frame sample for this sensor by locating: dish path and sensor -> frame_number
            target_sample = self.dataset_narrow_dict[split]["all_vids"].match((FF("dish_path")==frame_samples["pv"].dish_path) & (FF("sensor.name")==sensor)).first()
            sensor_frame_number = frame_samples[self.anchor_sensor].frame_number // (anchor_frame_rate/int(target_sample.metadata.frame_rate)) # frame_number in the target sensor inferenced from frame_number in the anchor sensor by difference in frame_rate
            frame_samples[sensor] = target_sample.frames[sensor_frame_number]
            # self.dataset_narrow_dict[split]["all_frames"].match((FF("dish_path")==frame_samples[self.anchor_sensor].dish_path) and (FF("sensor_name")==sensor) and (FF("frame_number")==frame_samples[self.anchor_sensor].frame_number)).first()
        # [print(f"{sen}: {fs}") for sen, fs in frame_samples.items()]        
        return frame_samples
    
    def load_predictions(self, split, predictions):
        """
        Load predictions to fiftyone dataset and evaluate the results.
        args:
            split: str. "train" or "val"
            predictions: a dict of a dict; {frame_filepath: {"class": int, "confidence": float, "logit": list of float}}
        """
        pred_filepath_list = list(predictions.keys())
        missed_prediction_filepath_list = []
        all_pred_Classification = []
        all_pred_classes = []
        all_gt_classes = []
        for video_sample in self.dataset_narrow_dict[split]["all_vids"].iter_samples(progress=True):
            all_pred_Classification.append([])
            for frame_no, frame_sample in video_sample.frames.items():
                all_gt_classes.append(frame_sample.step.step_no)
                if frame_sample.filepath not in pred_filepath_list:
                    missed_prediction_filepath_list.append(frame_sample.filepath)
                    all_pred_Classification[-1].append(None)
                    all_pred_classes.append(None)
                else:
                    all_pred_Classification[-1].append(fo.Classification(
                        label=task_spec[self.recipe_no][predictions[frame_sample.filepath]["class"]],
                        tags=[self.get_split_tag(self.fold_no, split)],
                        step_no=predictions[frame_sample.filepath]["class"],
                        confidence=predictions[frame_sample.filepath]["confidence"],
                        logits=predictions[frame_sample.filepath]["logits"]))
                    all_pred_classes.append(predictions[frame_sample.filepath]["class"])
        # write to FO
        self.dataset_narrow_dict[split]["all_vids"].set_values(f"frames.step_pred_{self.get_split_tag(self.fold_no, split)}", all_pred_Classification)
        
        # compute accuracy
        tmp_f = np.array(all_pred_classes)
        all_pred_classes = np.array(all_pred_classes)[tmp_f != None]
        all_gt_classes = np.array(all_gt_classes)[tmp_f != None]
        accuracy = np.sum(all_pred_classes == all_gt_classes) / len(all_gt_classes)
        self.accuracy_dict[self.fold_no][split] = accuracy
        
    def load_video_ann(self):
        group_slices = all_sensors.copy()
        if "microphone" in group_slices:
            group_slices.remove("microphone")
        all_naive_TDs = {gn: [] for gn in group_slices} # naively translated TemporalDetections for all videos
        all_processed_TDs = {gn: [] for gn in group_slices} # processed TemporalDetections for all videos
        
        for dish in self.dataset.iter_groups(group_slices=group_slices, progress=True):
            for sensor, sample in dish.items():
                
                # naively translate annotation file to TemporalDetections
                naive_TDs = annotation_file2TDs(sample)
                all_naive_TDs[sensor].append(naive_TDs)
                            
                # process the TemporalDetections to fit "no-gap & no-overlap" assumption
                all_processed_TDs[sensor].append(annotation_postprocess(naive_TDs, sample.metadata.total_frame_count))
                
        for sensor in group_slices:
            self.dataset.select_group_slices(sensor).set_values("step_raw", all_naive_TDs[sensor])
            self.dataset.select_group_slices(sensor).set_values("step", all_processed_TDs[sensor])
            
    def get_dataloaders(self, **kwargs_dataloader):
        """
        Get dataloaders for each split and return a dict of dataloaders.
        args:
            kwargs_dataloader: keyword arguments for torch.utils.data.DataLoader
        return:
            dataloaders: a dict of dataloaders. {split: dataloader}
        """
        self.dataloaders = {}
        for split in splits:
            self.dataloaders[split] = DataLoader(dataset=ExampleTorchDataset(self, split),
                                                 **kwargs_dataloader)
        return self.dataloaders
    
    def evaluate(self):
        # Frame Dataset cover all frames in the scope (recipe, sensor...)
        dataset_narrow = self.dataset.match(FF("recipe_no") == self.recipe_no).select_group_slices(self.anchor_sensor).to_frames()
        # avg accuracy over all folds for each split
        split_avg_accuracy = {}
        for split in splits:            
            split_avg_accuracy[split] = np.mean([self.accuracy_dict[fold_no][split] for fold_no in range(self.k_folds) ]).item()
        self.dataset.info[f"{self.k_folds}-folds average accuracy"] = split_avg_accuracy
        print("\n************************************************************************************************************************")
        print("************************************************************************************************************************")
        print(f"****************{self.k_folds}-fold validation accuracy: {split_avg_accuracy}******************")
        print("************************************************************************************************************************")
        print("************************************************************************************************************************\n")
        
    def load_forecasting_annotation(self, step_scope, recipe_scope=[0], sensor_scope=["pv"]):
        """
        return a FO Clip view and a Frame view that loaded forecasting annotation (path). Generate annotation and to disk if not exist.
        
        assume using _media_v000
        """
        
        # dataset_raw: whole dataset
        print("preparing dataset_raw...")
        new_dataset_name = "MILLYCookbook_FO_v016"
        if new_dataset_name in fo.list_datasets():
            dataset_raw = fo.load_dataset(new_dataset_name)
            fo_exist = True
        else:
            dataset_raw = self.dataset.clone(new_dataset_name, persistent=True)
            dataset_raw.default_skeleton = fo.KeypointSkeleton(labels=["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], edges=[[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]])
            fo_exist = False
            
        print("preparing dataset_videos...")            
        # dataset_videos: only pinwheel videos
        dataset_videos = dataset_raw.match(FF("recipe_no").is_in(recipe_scope)).select_group_slices(sensor_scope)
        print("preparing splits at dataset_videos...")            
        # split
        dataset_videos.untag_samples("val")
        dataset_videos.untag_samples("train")
        ratio_val = 0.1
        dataset_videos.take(int(ratio_val * len(dataset_videos)), seed=51).tag_samples("val")
        dataset_videos.match_tags("val", bool=False).tag_samples("train")
        dataset_videos.save()
        
        # add some patches to FO Dataset for this function at video level
        if not fo_exist:
            all_vid_frame_path_list = dataset_videos.values("frames.filepath", unwind=False)
            all_vid_new_frame_filepath_list = [] 
            all_vid_new_frame_dir_path_list = [] 
            for frame_path_list in all_vid_frame_path_list:
                new_frame_filepath_list = [] 
                new_frame_dir_path_list = [] 
                for frame_path in frame_path_list:
                    assert sensor_scope==["pv"], "only pv sensor is supported for now"
                    assert self.dataset_dir=='/z/dat/CookBook/MILLYCookbook_media_v000', "this part of code is adhoc for MILLYCookbook_media_v000"
                    tmp_name = frame_path.replace("pv/", "pv_frames/frame_")
                    # frame_no paths on the disk start from 0
                    frame_no_in_path = f"{int(tmp_name.split('frame_')[-1][:-4]) - 1:010d}" 
                    
                    new_filepath = os.path.join(os.path.dirname(tmp_name), "frame_"+frame_no_in_path+'.jpg')
                    new_frame_filepath_list.append(new_filepath)
                    
                    new_frame_dir_path_list.append(new_filepath[:-4])
                all_vid_new_frame_filepath_list.append(new_frame_filepath_list)
                all_vid_new_frame_dir_path_list.append(new_frame_dir_path_list)
            dataset_videos.set_values("frames.filepath", all_vid_new_frame_filepath_list)
            dataset_videos.set_values("frames.frame_info_dir", all_vid_new_frame_dir_path_list)
        # save view
        if "videos" in dataset_raw.list_saved_views():
            dataset_raw.delete_saved_view("videos")
        dataset_raw.save_view("videos", dataset_videos)
        
            
        print("preparing dataset_blobs...")
        # make blobs: blobs among all videos in dataset_videos
        # dataset_blobs = fu.build_blobs(dataset_videos, src_field_name="step")
        # narrow down to specific steps
        # dataset_blobs = dataset_videos.to_clips("step").match_frames(FF("step.step_no").is_in(step_scope))
        # match_clips_all_frames_in_scope = FF("frames").filter(FF("step.step_no").is_in(step_scope)).length() == FF("frames").length()
        # dataset_blobs = dataset_blobs.match(match_clips_all_frames_in_scope)
        dataset_blobs = dataset_videos.to_clips("step").match(FF("step.label").is_in([task_spec[0][step_no] for step_no in step_scope]))
        
        # save view
        if "blobs" in dataset_raw.list_saved_views():
            dataset_raw.delete_saved_view("blobs")
        dataset_raw.save_view("blobs", dataset_blobs)
        
        # make/load forecasting annotation if necessary
        print(f"==load frame level samples==")
        function_kwargs = {
            # "get_pose": {"return_type": "path", "write_fo": False},
            # "check_completion": {"read_type": "path", "return_type": "path", "write_fo": False},
            "forecasting_groundtruth": {"read_type": "path", "return_type": "path", "write_fo": True, "force_generate": False, "skip_generate": False}
        }
        for function, fargs in function_kwargs.items():
            fargs["dataset"] = dataset_blobs
            fargs["num_workers"] = os.cpu_count()
            fargs["function"] = function
            process = fu.MyProcess(**fargs)
            print(f"{function=} start")
            start_time = time.time()
            process.start_and_join()
            print(f"==frame level loading DONE in {time.time()-start_time:.2f}s==")
            print(f"{function=} done in {time.time()-start_time:.0f}s")
            
        print(f"DONE forecasting annoation generating for {step_scope=}")
        
        print("preparing dataset_clips...")        
        # make clips: check all completion and narrow down clips according to arguments
        # dataset_clips = dataset_videos.to_clips("step").match(FF("step.label").is_in([task_spec[0][step_no] for step_no in step_scope]))
        dataset_clips = dataset_blobs
        
        if "clips" in dataset_raw.list_saved_views():
            dataset_raw.delete_saved_view("clips")
        dataset_raw.save_view("clips", dataset_clips)
        print(f"{len(dataset_clips)=}\n{dataset_clips.count('frames')=}")
        
        print("preparing dataset_frames...")
        # dataset_frames: convert clips to frames dataset
        dataset_frames = dataset_clips.to_frames()
        if "frames" in dataset_raw.list_saved_views():
            dataset_raw.delete_saved_view("frames")
        dataset_raw.save_view("frames", dataset_frames)
        
        return dataset_raw, dataset_frames, None

def return_forecasting_dataset(step_scope):
    dataset_dir = '/z/dat/CookBook/MILLYCookbook_media_v000'
    benchmark = Benchmark(dataset_dir=dataset_dir, k_folds=3, sensors=['pv'], seed=51, force_reload=False, persistent=True, debug_mode=False, extract_frames=False, to_frames=True, visualize=False)
    return benchmark.load_forecasting_annotation(step_scope=step_scope)
        
if __name__ == "__main__":
    return_forecasting_dataset(step_scope=[9])  # 0,1,2,3,4,5,6,7,8,10,11,12,13




