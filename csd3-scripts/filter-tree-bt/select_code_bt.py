import sys
from pathlib import Path

"""usage is script INPUT OUTPUT_LOC"""

if __name__ == '__main__':
#    input_filename = sys.argv[1]
#    output_location = sys.argv[2]

    # check these are valid
    input_filename = Path(sys.argv[1])

    if not input_filename.exists():
        print("cannot find input file")
        exit(1)
    else:
        input_filename = input_filename.resolve()
        output_filename = input_filename.with_suffix('.en.filtered')
        print(f"output filename is {str(output_filename)}")

    # open file and read lines
    print(f"Reading input file {str(input_filename)}")
    with open(input_filename, 'r') as f:
        raw = f.readlines()
    print("file read")

    # read in top three for each id
    block_idx = None
    block_codes = []  # should contain three tuples when I'm done
    block_sents = []
    output = []  # I'm putting lists of tuples in this because two lists is stupid

    for line in raw:
        # split into fields
        idx, sentc, p1, p2 = line.split("|||")
        try:
            code, sent = sentc.split('<eoc>')
            if sent.endswith('</s>'):
                sent = sent[:-4]
        except ValueError:
            code = "NULL"
            sent = sentc

        if idx == block_idx:  # same block, just append until three in output
            # if there are three, then just keep adding
            if len(block_sents) < 3:
                block_codes.append(code)
                block_sents.append(sent)

            # if there are more than three, things get complicated
            else:
                if code == "NULL":  # don't care if code is a null
                    continue
                # get rid of any nulls that are there
                if "NULL" in block_codes:  # remove any NULLs and replace
                    # find index of last NULL
                    null_i = 2 - block_codes[::-1].index("NULL")
                    # remove null code and sent
                    block_codes.pop(null_i)
                    block_sents.pop(null_i)
                    # append new code
                    block_codes.append(code)
                    block_sents.append(sent)
                # if no nulls in block code, need to see if this is a new code
                elif code in block_codes:
                    continue  # don't want to add a less probably code
                elif len(set(block_codes)) == 3:
                    continue  # if all three already unique, I don't care
                else:  # if not in codes and all three not unique, then pop least likely dup
                    block_code_set = set(block_codes)
                    if len(block_code_set) == 1:  # all the same, so pop last
                        block_codes.pop()
                        block_sents.pop()
                        block_codes.append(code)
                        block_sents.append(sent)
                    else:  # iterate through and remove second dup
                        for i in range(3):
                            if block_codes[i] in block_code_set:
                                block_code_set.remove(block_codes[i])
                            else:
                                break
                        block_codes.pop(i)
                        block_sents.pop(i)
                        block_codes.append(code)
                        block_sents.append(sent)

        else:  # block changed, time to write
            # write out list
            if block_sents:
                output.append(list(zip(block_codes, block_sents)))
            # reset variables
            block_codes= []
            block_sents = []
            block_idx = idx
            block_codes.append(code)
            block_sents.append(sent)

# write out last output
output.append(list(zip(block_codes, block_sents)))

print("There are {} blocks".format(len(output)))

with open(output_filename, 'w') as out:
    for entry in output:
        for i in range(3):
            out.write(entry[i][0] + entry[i][1] + '\n')
print("script finished")
