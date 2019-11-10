import os
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import traceback

from .model import Alice, Bob, Eve
from .validation import validate


def train(args, trainloader, valloader, writer, logger, hp, hp_str):
    alice = Alice(hp).cuda()
    bob = Bob(hp).cuda()
    eve = Eve(hp).cuda()

    optim_e = torch.optim.Adam(
        eve.parameters(),
        lr=hp.train.adam.lr)
    optim_ab = torch.optim.Adam(
        list(alice.parameters()) + list(bob.parameters()),
        lr=hp.train.adam.lr)

    step = 0
    criterion = nn.L1Loss()
    try:
        alice.train(); bob.train(); eve.train()
        for epoch in itertools.count(0):
            if epoch % hp.log.validation == 0:
                with torch.no_grad():
                    validate(hp, args, alice, bob, eve, valloader, writer, step)

            loader = tqdm.tqdm(trainloader)
            for plainE, keyE, plainAB, keyAB in loader:
                plainE = plainE.cuda()
                keyE = keyE.cuda()
                plainAB = plainAB.cuda()
                keyAB = keyAB.cuda()

                # Eve
                optim_e.zero_grad()
                cipher = alice(plainE, keyE).detach()
                out_e = eve(cipher)
                loss_e = criterion(plainE, out_e)
                loss_e.backward()
                optim_e.step()
                loss_e_temp = loss_e.item()

                # Alice & Bob
                optim_ab.zero_grad()
                cipher = alice(plainAB, keyAB)
                out_e = eve(cipher)
                out_b = bob(cipher, keyAB)
                loss_e = criterion(plainAB, out_e)
                loss_b = criterion(plainAB, out_b)
                loss_ab = loss_b + (1. - loss_e).pow(2)
                loss_ab.backward()
                optim_ab.step()
                loss_b = loss_b.item()
                loss_ab = loss_ab.item()

                # logging
                step += 1
                tmp = max(loss_ab, loss_b, loss_e_temp)
                if tmp > 1e8 or math.isnan(tmp):
                    logger.error("loss exploded AB %f B %f E %f" % (loss_ab, loss_b, loss_e_temp))
                    raise Exception("Loss exploded")

                writer.log_train(loss_ab, loss_b, loss_e_temp, step)
                loader.set_description("AB %.04f B %.04f E %.04f step %d" % (loss_ab, loss_b, loss_e_temp, step))

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
