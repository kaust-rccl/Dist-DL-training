# File System Sanity Checks

This directory contains a set of self-contained sanity-check scripts used to verify user workspace access and dataset availability on Ibex.
You can run each script individually, or run all checks at once using the master script.

These checks are intended for lightweight validation before workshops, training runs, or automated job setups.

## Contents

| Script                  | Purpose                                                                                       |
|-------------------------|-----------------------------------------------------------------------------------------------|
| `file_system_access.sh` | Runs all checks: user space + Tiny ImageNet dataset.                                          |
| `user_space_access.sh`  | Verifies the user’s `/ibex/user/$USER` directory: path, permissions, write/read/delete tests. |
| `dataset_access.sh`     | Verifies access to `/ibex/reference/CV/tinyimagenet`: path, permissions, readability.         |

## Request access for `tinyimagenet` directory

Many training exercises use the TinyImageNet dataset stored in the shared data repository on IBEX.
Please request access before the workshop as follows:

- Log in to [https://my.ibex.kaust.edu.sa/](https://my.ibex.kaust.edu.sa/) using your IBEX username and password.

- From the top menu, go to Reference.

- In the search box, type “tinyimagenet”.

- Click Request Access next to the dataset entry.

- Wait for approval confirmation (usually processed within one working day).

Once approved, the dataset will be accessible under the shared reference directory:

````commandline
/ibex/reference/CV/tinyimagenet
````

## Running All Checks
To run every filesystem test at once:
```commandline
bash file_system_access.sh
```
### This script:

- Ensures the sub-scripts exist and are executable

- Runs `user_space_access.sh`

- Runs `dataset_access.sh`

- Prints each script’s native detailed output

## Running Checks Individually

### User Space Check

Validates your Ibex personal directory:

```commandline
bash user_space_access.sh
```

- This script performs:

- Path existence check

- RWX permission check

- Write test (creates file)

- Read test

- Delete test

- Summary with success/failure indicators

### Tiny ImageNet Dataset Check

Verifies access to the reference dataset:

```commandline
bash dataset_access.sh
```

- This script performs:
- Path existence check
- Read/list permission check
- Directory listing test
- Sample file readability (checks readable bytes)
- Summary with success/failure indicators

## Expected Output
### `user_space_access.sh` Summary
```commandline
[✓] Path exists: /ibex/user/<username>
[✓] Permissions OK (rwx)
[✓] Write test passed
[✓] Read test passed
[✓] Delete test passed
/ibex/user/<username> is accessible.
```
### `dataset_access.sh` Summary
```commandline
[✓] Path exists and is a directory
[✓] Permissions allow read/list
[✓] Readability test passed (listed contents and read a file)

Access to /ibex/reference/CV/tinyimagenet verified.
```
### file_system_access.sh
This simply prints the combined output from both scripts in sequence.