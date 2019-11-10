import torch


def validate(hp, args, alice, bob, eve, valloader, writer, step):
	alice.eval(); bob.eval(); eve.eval()

	correct_e = 0
	correct_b = 0
	for plain, key in valloader:
		plain = plain.cuda()
		key = key.cuda()
		cipher = alice(plain, key)
		outE = eve(cipher)
		outB = bob(cipher, key)
		correct_e += torch.sum(torch.abs(plain-outE)<1).item()
		correct_b += torch.sum(torch.abs(plain-outB)<1).item()

	acc_e = correct_e / len(valloader.dataset)
	acc_b = correct_b / len(valloader.dataset)

	writer.log_accuracy(acc_e, acc_b)

	alice.train(); bob.train(); eve.train()
