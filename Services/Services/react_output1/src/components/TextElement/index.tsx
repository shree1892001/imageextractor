import React from 'react';
import styled from 'styled-components';

interface TextElementProps {
  id: string;
  type: string;
  content: string;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  properties: {
    font: string;
    fontSize: number;
    color: number;
    flags: number;
  };
}

const TextElement: React.FC<TextElementProps> = ({ id, type, content, position, properties }) => {
  return (
    <StyledTextElement style={{ ...position, color: '#' + properties.color.toString(16) }}>
      {content}
    </StyledTextElement>
  );
};

export default TextElement;

const StyledTextElement = styled.div`
  position: absolute;
  font-family: ${({ properties }: { properties: any }) => properties.font};
  font-size: ${({ properties }: { properties: any }) => properties.fontSize}px;
  color: ${({ color }) => color};
`;