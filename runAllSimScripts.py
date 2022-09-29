import simAllEKFs, simAllUKFs, simAllEnKFs

# try:
#     print('Running EKF')
#     simAllEKFs.main()
# except:
#     print('EKF FAILED')

try:
    simAllUKFs.main()
except:
    print('UKF FAILED')

try:
    simAllEnKFs.main()
except:
    print('EnKF FAILED')