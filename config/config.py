class DefaultConfig(object):
    MAX_SEGMENT_NUM=20
    MAX_FRAME_NUM=200
    IN_CHANNEL=2048
    MAX_EPOCH=80
    BATCH_SIZE=10
    SIMILARITY_DIM=1000
    CONCEPT_DIM=300
    conv1_channel=512
    conv2_channel=256
    deconv1_channel=1024
    deconv2_channel=2048
    lr=1e-5

    TOP_PERCENT=0.02
    train_videos=[2,3,4]
    test_video=1
    gpu="0"

