from typing import List


class Vector3D:
    """
    https://carla.readthedocs.io/en/latest/python_api/#carla.Vector3D
    """
    def __init__(self, x: float = 0., y: float = 0., z: float = 0.) -> None:
        self.x = x
        self.y = y
        self.z = z


class Location(Vector3D):
    """
    https://carla.readthedocs.io/en/latest/python_api/#carla.Location
    """
    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        super().__init__(x, y, z)


class Rotation:
    """
    https://carla.readthedocs.io/en/latest/python_api/#carla.Rotation
    """
    def __init__(self, pitch: float = 0., yaw: float = 0., roll: float = 0.) -> None:
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    # def __str__(self):
    #     return f'CarlaRotation(pitch = {self.pitch}, yaw = {self.yaw}, roll = {self.roll})'


class Transform:
    """
    https://carla.readthedocs.io/en/latest/python_api/#carla.Transform
    """
    def __init__(self, location=Location(0, 0, 0), rotation=Rotation(0, 0, 0)) -> None:
        self.location = location
        self.rotation = rotation

    # def __str__(self):
    #     return f'CarlaTransform({self.location.__str__()}, {self.rotation.__str__()})'


class Actor:
    """
    https://carla.readthedocs.io/en/latest/python_api/#carlaactor
    """
    def __init__(self) -> None:
        self.attributes = dict()
        self.id = int()
        self.is_alive = bool()
        self.parent = Actor()
        self.semantic_tags: list[int] = list()
        self.type_id = str()

    """
        Methods
    """

    def destroy(self):
        return bool()

    """
        Getters 
    """

    def get_acceleration(self):
        """
        Returns the actor's 3D acceleration vector the client recieved during last tick. The method does not call the simulator.
        
        Return:
        ---
        - `carla.Vector3D` - m/s^2
        """
        return Vector3D()

    def get_angular_velocity(self):
        """
        Returns the actor's angular velocity vector the client recieved during last tick. The method does not call the simulator.
        
        Return:
        ---
        - `carla.Vector3D` - deg/s
        """
        return Vector3D()

    def get_location(self):
        """
        Returns the actor's angular velocity vector the client recieved during last tick. The method does not call the simulator.
        
        Return:
        ---
        - `carla.Location` - meters
        """
        return Location()

    def get_transform(self):
        """
        Returns the actor's angular velocity vector the client recieved during last tick. The method does not call the simulator.
        
        Return:
        ---
        - `carla.Location` - meters
        """
        return Transform()

    def get_velocity(self):
        """
        Returns the actor's angular velocity vector the client recieved during last tick. The method does not call the simulator.
        
        Return:
        ---
        - `carla.Location` - meters
        """
        return Vector3D()

    """
        Setters
    """

    def set_enable_gravity(self, enabled=True):
        """
        Enables or disables gravity for the actor. Default is True.
        """
        pass

    def set_location(self, location: Location):
        """
        Teleports the actor to a given location.
        """
        pass

    def set_transform(self, transform: Transform):
        """
        Teleports the actor to a given transform (location and rotation).
        """
        pass
