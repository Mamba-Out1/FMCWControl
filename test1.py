# import the Python wrapper
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import create_dict_from_sequence

# connect to an FMCW 60GHz radar sensor e.g. BGT60TR13C or BGT60UTR11AIP
with DeviceFmcw() as device:
    print("Radar SDK Version: " + get_version())
    print("UUID of board: " + device.get_board_uuid())
    print("Sensor: " + str(device.get_sensor_type()))

# A device instance is initialized with the default acquisition
    # sequence for its corresponding radar sensor. This sequence can be
    # simply fetched, analyzed or modified by the user.
    sequence = device.get_acquisition_sequence()

# Fetch a number of frames
    for frame_number in range(10):
        frame_contents = device.get_next_frame()

        for frame in frame_contents:
            num_rx = np.shape(frame)[0]

            # Do some processing with the obtained frame.
            # In this example we just dump it into the console
            print("Frame " + format(frame_number) + ", num_antennas={}".format(num_rx))

            for iAnt in range(num_rx):
                mat = frame[iAnt, :, :]
                print("Antenna", iAnt, "\n", mat)