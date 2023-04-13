import os
import requests
from tqdm import tqdm
import hashlib

# function to verify checksum of the downloaded file
def _md5Checksum(file_path):

    with open(file_path, "rb") as fh:

        m = hashlib.md5()

        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data)

        return m.hexdigest()


# main function
def download_array_name(dir=None):

    """Downloads a file from a given url and stores it in a local machine.

    Parameters
    ----------
    dir: str, optional
        The path of the file to be downloaded. If no path is provided, the default path will be used.

    Returns
    -------
    None
    """

    # define default url and file name

    # url = 'https://zenodo.org/record/7726336/files/full_dataset.tsv.gz.part-ak?download=1'
    # url = "https://zenodo.org/record/7677312/files/sim.py"
    url = "https://zenodo.org/record/7651129/files/covid19za_provincial_cumulative_timeline_vaccination.csv?download=1"
    filename = "array.txt"

    # check if a directory argument was given
    if dir is None:

        # find home directory
        dir = os.environ.get("HOME")

        # directories names
        module_name = "micromodelsimtest"
        folder_name = "data"

        # creat the full path
        dir_module = os.path.join(dir, module_name, folder_name)

    else:

        dir_module = dir

    # check if path already exits.
    if not os.path.exists(dir_module):

        os.makedirs(dir_module)

    dir_filename = os.path.join(dir_module, filename)

    # checks if the string dir_filename exists as a path
    if os.path.exists(dir_filename):

        print("File already exists")

    else:
        try:

            # create a request resnponse object
            r = requests.get(url, stream=True)
            # print(r.headers)


            with open(dir_filename, "wb") as f:

                total_length = int(r.headers.get("content-length"))

                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=total_length / 1024,
                    unit="KB",
                ):

                    if chunk:
                        f.write(chunk)

            # creates a hash from the file downloaded in the local machine
            md5_hash = _md5Checksum(dir_filename)

            # compare the local hash code with the hash code in the request object header attribute
            if md5_hash == r.headers.get("Content-MD5"):

                print("File checksum verified")

            else:

                print("File checksum unsuccesful")

        except Exception as e:

            print("Something went wrong: {}".format(e))

        else:
            print("Successfully downloaded file!")


# test data below

# path = 'https://zenodo.org/record/7726336/files/full_dataset.tsv.gz.part-ak?download=1'
# path = 'https://zenodo.org/record/7677312/files/sim.py'

download_array_name()
# download_array_name("/home/paul/luuujjjnnggg")

# dowdocstring = download_array_name.__doc__
# print(dowdocstring)