import carla
tr = carla.Transform(
    carla.Location(1.0, 1.0, 1.0),
    carla.Rotation(0.0, 90.0, 0.0),
)
print(str(tr.transform(carla.Location(5.0, 3.0, 4.0))))