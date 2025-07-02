import React from 'react';
import styled from 'styled-components';

const TextContainer = styled.div`
  position: relative;
`;

const TextElement = styled.span`
  position: absolute;
  left: ${({ x }) => `${x}px`};
  top: ${({ y }) => `${y}px`};
  font-size: ${({ fontSize }) => `${fontSize}px`};
  font-family: ${({ font }) => font};
  color: ${({ color }) => `rgb(${color})`};
`;

interface TextElementProps {
    id: string;
    x: number;
    y: number;
    width: number;
    height: number;
    content: string;
    font: string;
    fontSize: number;
    color: number;
    flags: number;
}

const TextElementComponent: React.FC<TextElementProps> = ({ id, x, y, width, height, content, font, fontSize, color }) => {
    return (
        <TextElement x={x} y={y} fontSize={fontSize} font={font} color={color}>
            {content}
        </TextElement>
    );
};

const TextPage1: React.FC = () => {
    const textElementsData = [
        // ... (Your text element data here)
    ];
    return (
        <TextContainer>
            {textElementsData.map((element) => (
                <TextElementComponent key={element.id} {...element} />
            ))}
        </TextContainer>
    );
};

export default TextPage1;
