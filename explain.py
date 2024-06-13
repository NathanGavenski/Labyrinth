from functools import partial
from typing import Any, Callable, Dict
import tkinter as tk
import gym

from pytorch_grad_cam import XGradCAM as GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from torch import nn
from torchvision import transforms
from PIL import ImageTk, Image
import numpy as np
from imitation_datasets.dataset import BaselineDataset
from tqdm import tqdm

from benchmark.methods.bc import BC
from src import maze


class RLGUI:
    def __init__(self, master, transform):
        self.master = master
        self.transform = transform
        self.method = None
        self.selected_model = tk.StringVar(value="Resnet")
        self.font = ("Helvetica", 12, "bold")
        self.params = {
            "shape": (5, 5),
            "screen_width": 600,
            "screen_height": 600,
            "visual": True,
        }
        self.setup_gui()

    def setup_gui(self):
        self.master.title("Model Loader")

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        window_width = int(screen_width * 0.15)
        window_height = int(screen_height * 0.10)
        self.master.geometry(f"{window_width}x{window_height}")

        label = tk.Label(self.master, text="Input weight path")
        label.pack(pady=(window_height * 0.05, 0), padx=(window_width * 0.05, window_width * 0.05))

        self.entry = tk.Entry(self.master)
        self.entry.pack(
            pady=(window_height * 0.05, window_height * 0.05),
            padx=(window_width * 0.05, window_width * 0.05),
            fill=tk.BOTH,
            expand=True
        )
        self.entry.insert(0, "./tmp/bc/Maze/400")

        radio_frame = tk.Frame(self.master)
        radio_frame.pack(pady=(0, window_height * 0.05))

        cnn_button = tk.Radiobutton(
            radio_frame,
            text="CNN",
            variable=self.selected_model,
            value="CNN"
        )
        cnn_button.pack(side=tk.LEFT)

        resnet_button = tk.Radiobutton(
            radio_frame,
            text="Resnet",
            variable=self.selected_model,
            value="Resnet"
        )
        resnet_button.pack(side=tk.LEFT)

        att_button = tk.Radiobutton(
            radio_frame,
            text="Att",
            variable=self.selected_model,
            value="Att"
        )
        att_button.pack(side=tk.LEFT)

        load_button = tk.Button(self.master, text="Load", command=self.load_model)
        load_button.pack(pady=(0, window_height * 0.05))

    def load_model(self):
        env = gym.make("Maze-v0", **self.params)
        bc = BC(env, config_file=f"./configs/{self.selected_model.get().lower()}.yaml")
        path = self.entry.get().split('/')
        bc = bc.load(path=f"{'/'.join(path[:-1])}/", name=path[-1])

        self.method = Method(bc, env, self.transform)
        self.create_explainable_window()

    def create_explainable_window(self):
        self.explainable_window = tk.Toplevel(self.master)
        self.explainable_window.title(self.entry.get())

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.window_width = int(screen_width * 0.25)
        self.window_height = int(screen_height * 0.75)
        self.explainable_window.geometry(f"{self.window_width}x{self.window_height}")

        maze_label = tk.Label(self.explainable_window, text="Maze path:")
        maze_label.grid(
            row=0,
            column=0,
            pady=(self.window_height * 0.025, 0),
            padx=(self.window_width * 0.05, 0),
            sticky="w"
        )

        self.maze_entry = tk.Entry(self.explainable_window)
        self.maze_entry.grid(
            row=0,
            column=1,
            columnspan=3,
            pady=(self.window_height * 0.025, 0),
            sticky="ew"
        )
        self.maze_entry.insert(0, "./src/environment/mazes/mazes5/test/2562125594538930084.txt")

        maze_load = tk.Button(self.explainable_window, text="Load", command=self.load_env)
        maze_load.grid(
            row=0,
            column=4,
            pady=(self.window_height * 0.025, 0),
            padx=(0, self.window_width * 0.05),
            sticky="w"
        )

        self.state_label = tk.Canvas(self.explainable_window, width=224, height=224)
        self.state_label.grid(
            row=2,
            column=1,
            padx=(0, self.window_width * 0.05),
            sticky="w"
        )

        self.saliency_label = tk.Canvas(self.explainable_window, width=224, height=224)
        self.saliency_label.grid(
            row=2,
            column=2,
            padx=(0, self.window_width * 0.05),
            sticky="ew"
        )

        self.close_label = tk.Canvas(self.explainable_window, width=224, height=224)
        self.close_label.grid(row=2, column=3, sticky="e")
        self.resize(self.explainable_window)

    def resize(self, window):
        window.update_idletasks()
        width = window.winfo_reqwidth()
        height = window.winfo_reqheight()
        window.geometry(f"{width}x{height}")

    def load_env(self):
        self.method.load(self.maze_entry.get())
        self.render_env()

    def render_env(self, action: int = None):
        if self.method.state is not None:
            # Row 1
            label = tk.Label(self.explainable_window, text="current state:")
            label.grid(
                row=1,
                column=1,
                pady=(self.window_height * 0.05, 0),
                padx=(0, self.window_width * 0.05),
                sticky="ew"
            )
            label = tk.Label(self.explainable_window, text="saliency map:")
            label.grid(
                row=1,
                column=2,
                pady=(self.window_height * 0.05, 0),
                padx=(0, self.window_width * 0.05),
                sticky="ew"
            )
            label = tk.Label(self.explainable_window, text="retrieval state:")
            label.grid(
                row=1,
                column=3,
                pady=(self.window_height * 0.05, 0),
                sticky="ew"
            )

            # Row 2
            model_predictions, dict_predict = self.method.predict()
            image = self.method.revert(self.method.state)
            self.state_tk_image = ImageTk.PhotoImage(image)
            self.state_label.create_image(0, 0, image=self.state_tk_image, anchor="nw")

            image = self.method.gradcam(np.argmax(model_predictions) if action is None else action)
            self.saliency_tk_image = ImageTk.PhotoImage(image)
            self.saliency_label.create_image(0, 0, image=self.saliency_tk_image, anchor="nw")

            image = self.method.retrieve()
            self.close_tk_image = ImageTk.PhotoImage(image)
            self.close_label.create_image(0, 0, image=self.close_tk_image, anchor="nw")

            # Row 3
            prediction_label = tk.Label(
                self.explainable_window,
                text="Policy predictions",
                font=self.font
            )
            prediction_label.grid(
                row=3,
                column=0,
                columnspan=5,
                sticky="ew",
                pady=(self.window_height * 0.025, 0)
            )

            # Row 4
            predictions_text = "    ".join(
                [f"{i}: {value}%" for i, value in dict_predict.items()]
            )
            predictions = tk.Label(self.explainable_window, text=predictions_text, font=self.font)
            predictions.grid(
                row=4,
                column=0,
                columnspan=5,
                sticky="ew",
            )

            # Row 5
            action_label = tk.Label(
                self.explainable_window,
                text="Actions:",
                font=self.font
            )
            action_label.grid(
                row=5,
                column=0,
                columnspan=5,
                sticky="w",
                pady=(self.window_height * 0.025, 0)
            )

            # Row 6
            first_action = tk.Button(
                self.explainable_window,
                text="1st (UP)",
                command=self.step(0)
            )
            first_action.grid(row=6, column=0, sticky="ew")
            second_action = tk.Button(
                self.explainable_window,
                text="2nd (RIGHT)",
                command=self.step(1)
            )
            second_action.grid(row=6, column=1, sticky="ew")
            third_action = tk.Button(
                self.explainable_window,
                text="3rd (DOWN)",
                command=self.step(2)
            )
            third_action.grid(row=6, column=2, sticky="ew")
            forth_action = tk.Button(
                self.explainable_window,
                text="4th (LEFT)",
                command=self.step(3)
            )
            forth_action.grid(row=6, column=3, sticky="ew")
            argmax_action = tk.Button(
                self.explainable_window,
                text=f"argmax ({np.argmax(model_predictions) + 1})",
                command=self.step(np.argmax(model_predictions))
            )
            argmax_action.grid(row=6, column=4, sticky="ew")

            # Row 7
            gradcam_label = tk.Label(
                self.explainable_window,
                text="GradCAM:",
                font=self.font
            )
            gradcam_label.grid(
                row=7,
                column=0,
                columnspan=5,
                sticky="w",
                pady=(self.window_height * 0.025, 0)
            )

            # Row 8
            first_action = tk.Button(
                self.explainable_window,
                text="1st (UP)",
                command=self.gradcam(0)
            )
            first_action.grid(row=8, column=0, sticky="ew")
            second_action = tk.Button(
                self.explainable_window,
                text="2nd (RIGHT)",
                command=self.gradcam(1)
            )
            second_action.grid(row=8, column=1, sticky="ew")
            third_action = tk.Button(
                self.explainable_window,
                text="3rd (DOWN)",
                command=self.gradcam(2)
            )
            third_action.grid(row=8, column=2, sticky="ew")
            forth_action = tk.Button(
                self.explainable_window,
                text="4th (LEFT)",
                command=self.gradcam(3)
            )
            forth_action.grid(row=8, column=3, sticky="ew")
            argmax_action = tk.Button(
                self.explainable_window,
                text=f"argmax ({np.argmax(model_predictions) + 1})",
                command=self.gradcam(np.argmax(model_predictions))
            )
            argmax_action.grid(row=8, column=4, sticky="ew")

            # Row 9
            previous_action = tk.Button(
                self.explainable_window,
                text="<<",
                command=partial(self.method.reverse, callback=self.render_env)
            )
            previous_action.grid(row=9, column=1, sticky="ew")
            forward_action = tk.Button(
                self.explainable_window,
                text=">>",
                command=partial(self.method.forward, callback=self.render_env)
            )
            forward_action.grid(row=9, column=3, sticky="ew")

        self.resize(self.explainable_window)

    def step(self, action: int) -> Callable[[int], None]:
        return partial(self.method.step, action=action, callback=self.render_env)

    def gradcam(self, action: int) -> Callable[[int], Image.Image]:
        return partial(self.render_env, action=action)


class Method:

    REVERSE = {0: 2, 1: 3, 2: 0, 3: 1}

    def __init__(
        self,
        method: BC,
        env: gym.Env,
        transform: Callable[[torch.tensor], torch.tensor]
    ) -> None:
        self.train_dataset = BaselineDataset(
            "NathanGavenski/imagetrain",
            source="hf",
            hf_split="shortest_route",
            transform=transforms.Resize(64)
        )

        self.method = method
        self.env = env
        self.transform = transform
        self.revert = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.ToPILImage()
        ])

        self.state = np.array([])
        self.actions = []
        self.index = -1
        self.cam = GradCAM(
            model=self.method.policy,
            target_layers=[self.method.policy[0].model.layer4[-1]]
        )

        self.get_retrieval_info()

    def get_features(self, x):
        with torch.no_grad():
            x = self.method.policy[0].model.conv1(x)
            x = self.method.policy[0].model.bn1(x)
            x = self.method.policy[0].model.relu(x)
            x = self.method.policy[0].model.maxpool(x)
            x = self.method.policy[0].model.layer1(x)
            x = self.method.policy[0].model.layer2(x)
            x = self.method.policy[0].model.layer3(x)
            x = self.method.policy[0].model.layer4(x)
            features = self.method.policy[0].model.avgpool(x)
            features = features.squeeze(2, 3)
            x = self.method.policy[1](features)
            return torch.argmax(x, dim=1).item(), features.squeeze().detach().numpy()

    def get_retrieval_info(self) -> None:
        retrieval_data = {}
        for state, action, *_ in tqdm(self.train_dataset):
            pred_action, features = self.get_features(state[None])
            dict_key = tuple(state.swapaxes(0, 2).swapaxes(0, 1).reshape(-1).tolist())
            retrieval_data[dict_key] = {
                "action": pred_action,
                "gt_action": action,
                "features": features
            }
        self.retrieval_data = retrieval_data

    def retrieve(self) -> None:
        pred_action, feature = self.get_features(self.transform_state(self.state))
        features = [value["features"] for value in self.retrieval_data.values()]
        features = np.array(features)

        distances = np.argmin(np.abs(features - feature).mean(axis=1), axis=0)
        closest_state = list(self.retrieval_data.keys())[distances]
        closest_state = torch.tensor(closest_state).reshape(64, 64, 3).numpy()
        return self.revert(closest_state)

    def load(self, path: str) -> None:
        self.env.close()
        self.state = self.env.load(path)

    def transform_state(self, state):
        state = self.transform(self.state)
        if len(state.shape) < 4:
            state = state[None]
        return state

    def predict(self) -> Dict[int, float]:
        if self.state is None:
            return None
        state = self.transform_state(self.state)

        self.method.policy.eval()
        self.method.policy.zero_grad()
        with torch.no_grad():
            predictions = self.method.forward(state)[0]
        predictions = nn.Softmax(dim=0)(predictions) * 100
        predictions = predictions.tolist()
        dict_predict = {i + 1: round(value, 2) for i, value in enumerate(predictions)}
        return predictions, dict_predict

    def gradcam(self, target: int) -> Image.Image:
        self.method.policy.zero_grad()
        state = self.transform_state(self.state)
        targets = [ClassifierOutputTarget(target)]
        grayscale_cam = self.cam(
            input_tensor=state,
            targets=targets,
        )
        grayscale_cam = grayscale_cam[0, :]
        grad_cam = show_cam_on_image(
            np.array(transforms.ToPILImage()(state[0])) / 255,
            grayscale_cam,
            image_weight=0.5
        )
        return self.revert(grad_cam)

    def step(self, action: int, callback: Callable[[Any], Any]) -> None:
        if self.index == -1:
            self.actions.append(action)
        else:
            self.actions[self.index] = action

        self.state, *_ = self.env.step(action)
        callback()

    def reverse(self, callback: Callable[[Any], Any]) -> None:
        action = self.REVERSE[self.actions[self.index]]
        self.index -= 1
        self.state, *_ = self.env.step(action)
        callback()

    def forward(self, callback: Callable[[Any], Any]) -> None:
        if self.index == -1:
            return None

        self.index += 1
        action = self.actions[self.index]
        self.state, *_ = self.env.step(action)
        callback()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
    ])

    root = tk.Tk()
    app = RLGUI(root, transform=transform)
    root.mainloop()
