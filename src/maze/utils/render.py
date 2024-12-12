"""Module for rendering the environment."""
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from typing import List

import pygame

from .colors import Colors


class RenderUtils:
    """Render class responsible for rendering the environment.

    Args:
        shape (list[int]): (x, y) size for maze.
        viewer (Viewer, optional): viewer to render environment. Defaults to None.
        screen_info (list[int], optional): size of the viewer screen. Defaults to None.

    Raises:
        Exception: screen_info and viewer can not be None at the same time.
    """

    start = None
    end = None

    def __init__(
        self,
        shape: List[int],
        viewer: pygame.Surface = None,
        screen_info: List[int] = None,
    ) -> None:
        if viewer is None and screen_info is None:
            raise ValueError("screen_info and viewer can not be None at the same time.")

        self.shape = shape

        if screen_info is None:
            screen_h = viewer.get_height()
            screen_w = viewer.get_width()
        else:
            screen_w, screen_h = screen_info

        if viewer is not None:
            self.viewer = viewer
            self.viewer.fill(Colors.WHITE.value)

        width, height = shape
        self.tile_h = screen_h / height
        self.tile_w = screen_w / width

    def redraw(self) -> Self:
        """Redraws the environment.

        Returns:
            Self: return instance.
        """
        self.viewer.fill(Colors.WHITE.value)
        return self

    def draw_walls(
        self,
        maze: List[int],
    ) -> Self:
        """Renders walls for the maze.

        Args:
            maze (list[int]): Maze walls from _generate().

        Returns:
            self (self): return instance.
        """
        for x, tiles in enumerate(maze):
            if self.shape[0] * 2 > x > 0:
                for y, tile in enumerate(tiles):
                    if tile == 1 and self.shape[0] * 2 > y > 0:
                        start_pos, end_pos = None, None
                        if x % 2 == 0 and (y % 2 != 0 or y == 1):
                            # horizontal wall
                            _y = x // 2
                            _x = y // 2 + 1
                            start_pos = ((_x - 1) * self.tile_w, _y * self.tile_h)
                            end_pos = (_x * self.tile_w, _y * self.tile_h)
                        elif x % 2 > 0:
                            # vertical wall
                            _y = x // 2 + 1
                            _x = y // 2
                            start_pos = (_x * self.tile_w, (_y - 1) * self.tile_h),
                            end_pos = (_x * self.tile_w, _y * self.tile_h)
                        if start_pos is not None and end_pos is not None:
                            pygame.draw.line(
                                self.viewer, Colors.BLACK.value, start_pos, end_pos, 2
                            )
        return self

    def draw_agent(self, agent: List[int]) -> Self:
        """Renders agent for the maze. Sets agent_transition.

        Args:
            agent (list[int]): agent (x, y) position.

        Returns:
            self (self): return instance.
        """
        left = agent[1] * self.tile_w
        right = (agent[1] + 1) * self.tile_w
        bottom = agent[0] * self.tile_h
        top = (agent[0] + 1) * self.tile_h
        agent = [
            (left + self.tile_w // 2, bottom),
            (left, top - self.tile_h // 2),
            (right - self.tile_w // 2, top),
            (right, bottom + self.tile_h // 2)
        ]
        pygame.draw.polygon(self.viewer, Colors.GREEN.value, agent)
        return self

    def draw_end(self, end: List[int]) -> Self:
        """Renders end tile. Sets end position.

        Args:
            end (list[int]): end (x, y) position.

        Returns:
            self (self): return instance.
        """
        self.end = end
        rect = pygame.Rect(
            end[1] * self.tile_w,
            end[0] * self.tile_h,
            self.tile_w + 2,
            self.tile_h + 2
        )
        pygame.draw.rect(self.viewer, Colors.BLUE.value, rect)
        return self

    def draw_start(self, start: List[int]) -> Self:
        """Renders start tile. Sets start position.

        Args:
            start (list[int]): start (x, y) position.

        Returns:
            self (self): return instance.
        """
        self.start = start
        rect = pygame.Rect(
            start[1] * self.tile_w,
            start[0] * self.tile_h,
            self.tile_w + 2,
            self.tile_h + 2
        )
        pygame.draw.rect(self.viewer, Colors.RED.value, rect)
        return self

    def draw_mask(self, mask: List[List[int]]) -> Self:
        """Renders mask.

        Args:
            mask (list[list[int]]): mask based on agent position.

        Returns:
            self (self): return instance.
        """
        for y, tiles in enumerate(mask):
            if self.shape[0] * 2 > y > 0:
                for x, tile in enumerate(tiles):
                    _x = x // 2
                    _y = y // 2

                    if (_y, _x) in [self.start, self.end]:
                        continue

                    if (
                        tile == 1
                        and self.shape[1] * 2 > x > 0
                        and (x % 2 != 0 and y % 2 != 0)
                    ):
                        left = _x * self.tile_w
                        bottom = _y * self.tile_h
                        rect = pygame.Rect(left, bottom, self.tile_w + 2, self.tile_h + 2)
                        pygame.draw.rect(self.viewer, Colors.BLACK.value, rect)
        return self

    def draw_key(self, key: List[int]) -> Self:
        """Renders key.

        Args:
            key (list[int]): key (x, y) position.

        Returns:
            self (self): return instance.
        """
        if key is None:
            return self
        key_y, key_x = key
        rect = pygame.Rect(
            key_x * self.tile_w + self.tile_w * 0.25,
            key_y * self.tile_h + self.tile_h * 0.25,
            self.tile_w * 0.5,
            self.tile_h * 0.5,
        )
        pygame.draw.rect(self.viewer, Colors.GOLD.value, rect)
        return self

    def draw_door(self, door: List[int]) -> Self:
        """Renders door.

        Args:
            door (list[int]): door (x, y) position.

        Returns:
            self (self): return instance.
        """
        if door is None:
            return self
        door_y, door_x = door
        rect = pygame.Rect(
            door_x * self.tile_w,
            door_y * self.tile_h,
            self.tile_w + 2,
            self.tile_h + 2,
        )
        pygame.draw.rect(self.viewer, Colors.BROWN.value, rect)
        return self

    def draw_ice_floors(self, ice_floors: List[int]) -> Self:
        """Render ice floors.

        Args:
            ice_floors (list(int)): ice floors (x, y) positions.

        Returns:
            self (self): return instance.
        """
        for ice_floor in ice_floors:
            y, x = ice_floor
            rect = pygame.Rect(
                x * self.tile_w,
                y * self.tile_h,
                self.tile_w + 2,
                self.tile_h + 2,
            )
            pygame.draw.rect(self.viewer, Colors.ICE.value, rect)
        return self
