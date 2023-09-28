try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from gym.envs.classic_control import rendering

from src.maze.utils import Colors


class RenderUtils:
    """Render class responsible for rendering the environment.

    Args:
        shape (list[int]): (x, y) size for maze.
        viewer (Viewer, optional): viewer to render environment. Defaults to None.
        screen_info (list[int], optional): size of the viewer screen. Defaults to None.

    Raises:
        Exception: screen_info and viewer can not be None at the same time.
    """

    def __init__(
        self,
        shape: list[int],
        viewer: rendering.Viewer = None,
        screen_info: list[int] = None,
    ) -> None:
        if viewer is None and screen_info is None:
            raise Exception("screen_info and viewer can not be None at the same time.")

        self.shape = shape

        if screen_info is None:
            self.screen_h = viewer.height
            self.screen_w = viewer.width
        else:
            self.screen_w, self.screen_h = screen_info

        if viewer is not None:
            self.viewer = viewer
        else:
            self.viewer = rendering.Viewer(self.screen_w, self.screen_h)

        w, h = shape
        self.tile_h = self.screen_h / h
        self.tile_w = self.screen_w / w

        self.start = None
        self.end = None
        self.agent_transition = None

    def draw_walls(
        self,
        maze: list[int],
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
                        if x % 2 == 0 and (y % 2 != 0 or y == 1):  # horizontal wall
                            _y = x // 2
                            _x = y // 2 + 1
                            line = rendering.Line(
                                ((_x - 1) * self.tile_w, _y * self.tile_h),
                                (_x * self.tile_w, _y * self.tile_h)
                            )
                            line.set_color(*Colors.BLACK.value)
                            self.viewer.add_geom(line)
                        elif x % 2 > 0:  # vertical wall
                            _y = x // 2 + 1
                            _x = y // 2
                            line = rendering.Line(
                                (_x * self.tile_w, (_y - 1) * self.tile_h),
                                (_x * self.tile_w, _y * self.tile_h)
                            )
                            line.set_color(*Colors.BLACK.value)
                            self.viewer.add_geom(line)
        return self

    def draw_agent(self, agent: list[int]) -> Self:
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
        agent = rendering.FilledPolygon([
            (left + self.tile_w // 2, bottom),
            (left, top - self.tile_h // 2),
            (right - self.tile_w // 2, top),
            (right, bottom + self.tile_h // 2)
        ])
        self.agent_transition = rendering.Transform()
        agent.add_attr(self.agent_transition)
        agent.set_color(*Colors.GREEN.value)
        self.viewer.add_geom(agent)
        return self

    def draw_end(self, end: list[int]) -> Self:
        """Renders end tile. Sets end position.

        Args:
            end (list[int]): end (x, y) position.

        Returns:
            self (self): return instance.
        """
        self.end = end

        left = end[1] * self.tile_w
        right = (end[1] + 1) * self.tile_w
        top = end[0] * self.tile_h
        bottom = (end[0] + 1) * self.tile_h
        end = rendering.FilledPolygon([
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom)
        ])
        end.set_color(*Colors.BLUE.value)
        self.viewer.add_geom(end)
        return self

    def draw_start(self, start: list[int]) -> Self:
        """Renders start tile. Sets start position.

        Args:
            start (list[int]): start (x, y) position.

        Returns:
            self (self): return instance.
        """
        self.start = start

        left = start[1] * self.tile_w
        right = (start[1] + 1) * self.tile_w
        top = start[0] * self.tile_h
        bottom = (start[0] + 1) * self.tile_h
        start = rendering.FilledPolygon([
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom)
        ])
        start.set_color(*Colors.RED.value)
        self.viewer.add_geom(start)
        return self

    def draw_mask(self, mask: list[list[int]]) -> Self:
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

                    if tile == 1 and \
                        self.shape[1] * 2 > x > 0 and \
                            (x % 2 != 0 and y % 2 != 0):

                        left = _x * self.tile_w
                        right = (_x + 1) * self.tile_w
                        bottom = _y * self.tile_h
                        top = (_y + 1) * self.tile_h
                        mask = rendering.FilledPolygon([
                            (left, bottom),
                            (left, top),
                            (right, top),
                            (right, bottom)
                        ])
                        mask.set_color(*Colors.BLACK.value)
                        self.viewer.add_onetime(mask)
        return self

    def draw_key(self, key: list[int]) -> Self:
        """Renders key.

        Args:
            key (list[int]): key (x, y) position.

        Returns:
            self (self): return instance.
        """
        key_y, key_x = key
        left = key_x * self.tile_w + self.tile_w * 0.25
        right = (key_x + 1) * self.tile_w - self.tile_w * 0.25
        bottom = key_y * self.tile_h + self.tile_h * 0.25
        top = (key_y + 1) * self.tile_h - self.tile_h * 0.25

        key_rendering = rendering.FilledPolygon([
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom)
        ])
        key_rendering.set_color(*Colors.GOLD.value)
        self.viewer.add_onetime(key_rendering)
        return self

    def draw_door(self, door: list[int]) -> Self:
        """Renders door.

        Args:
            door (list[int]): door (x, y) position.

        Returns:
            self (self): return instance.
        """
        door_y, door_x = door
        left = door_x * self.tile_w
        right = (door_x + 1) * self.tile_w
        bottom = door_y * self.tile_h
        top = (door_y + 1) * self.tile_h

        door_rendering = rendering.FilledPolygon([
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom)
        ])
        door_rendering.set_color(*Colors.BROWN.value)
        self.viewer.add_onetime(door_rendering)
        return self

    def draw_ice_floors(self, ice_floors: list[int]) -> Self:
        """Render ice floors.

        Args:
            ice_floors (list(int)): ice floors (x, y) positions.

        Returns:
            self (self): return instance.
        """
        for ice_floor in ice_floors:
            y, x = ice_floor
            left = x * self.tile_w
            right = (x + 1) * self.tile_w
            bottom = y * self.tile_h
            top = (y + 1) * self.tile_h

            ice_rendering = rendering.FilledPolygon([
                (left, bottom),
                (left, top),
                (right, top),
                (right, bottom)
            ])
            ice_rendering.set_color(*Colors.ICE.value)
            self.viewer.add_onetime(ice_rendering)
        return self
