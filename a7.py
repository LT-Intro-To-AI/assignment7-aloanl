from neural import NeuralNet


def run_xor():
    xor_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]

    net = NeuralNet(2, 1, 1)
    net.train(xor_data, iters=500000)

    print(net.test_with_expected(xor_data))


def run_vote_prediction():
    voter_data = [
        ([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]),
        ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]),
        ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]),
        ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]),
        ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]),
        ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0]),
    ]

    test_voters = [
        [1.0, 1.0, 1.0, 0.1, 0.1],
        [0.5, 0.2, 0.1, 0.7, 0.7],
        [0.8, 0.3, 0.3, 0.3, 0.8],
        [0.8, 0.3, 0.3, 0.8, 0.3],
        [0.9, 0.8, 0.8, 0.3, 0.6],
    ]

    voter_net = NeuralNet(5, 80, 1)

    voter_net.train(voter_data, iters=15000)

    # print(voter_net.test_with_expected(voter_data))
    for test_voter in test_voters:
        print(f"{test_voter} is evaluated as {voter_net.evaluate(test_voter)}")


run_vote_prediction()
