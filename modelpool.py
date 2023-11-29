clean_model_path = {
    'fashion': [
        '/path/to/last.pth',
        '/path/to/last.pth',
        # ...
    ],
    'cifar10': [
        '/path/to/last.pth',
        '/path/to/last.pth',
        # ...
    ],
    'cifar100': [
        '/path/to/last.pth',
        '/path/to/last.pth',
        # ...
    ]
}

watermark_model_path = {
    # from pe-trained
    'ssw-p': {
        'fashion': [
            '/path/to/last.pth',
            '/path/to/last.pth',
            # ...
        ],
        'cifar10': [
            '/path/to/last.pth',
            '/path/to/last.pth',
            # ...
        ],
        'cifar100': [
            '/path/to/last.pth',
            '/path/to/last.pth',
            # ...
        ]
    },
    # from scratch
    'ssw-s': {
        'fashion': [
            '/path/to/last.pth',
            '/path/to/last.pth',
            # ...
        ],
        'cifar10': [
            '/path/to/last.pth',
            '/path/to/last.pth',
            # ...
        ],
        'cifar100': [
            '/path/to/last.pth',
            '/path/to/last.pth',
            # ...
        ]
    }
}
